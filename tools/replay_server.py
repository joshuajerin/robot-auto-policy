"""Raindrop Workshop replay server for the local AutoResearch dry-run agent."""

from __future__ import annotations

import json
import html
import mimetypes
import os
import re
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.openai_client import DEFAULT_MODEL
from core.autoresearch_loop import RAINDROP_EVENT_NAME, run_traced_dry_research_loop


PORT = 61020
COMMAND = "python tools/replay_server.py"
INPUT = {"experiments": "number"}
PREFILL_FROM_TRACE = {"experiments": "properties.experiments"}
MODELS = ["deterministic-local", os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)]


class ReplayHandler(BaseHTTPRequestHandler):
    server_version = "RoboGenesisReplay/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/artifact-file":
            self._send_artifact_file(parsed.query)
            return
        if parsed.path == "/artifact-video":
            self._send_artifact_video_page(parsed.query)
            return
        if parsed.path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "eventName": RAINDROP_EVENT_NAME,
                    "port": PORT,
                    "cwd": str(REPO_ROOT),
                    "command": COMMAND,
                    "input": INPUT,
                    "prefillFromTrace": PREFILL_FROM_TRACE,
                    "models": MODELS,
                    "artifactFileEndpoint": "/artifact-file?path=<repo-relative-path>",
                    "artifactVideoEndpoint": "/artifact-video?path=<repo-relative-mp4>",
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/replay":
            self._send_json(404, {"status": "error", "message": "not_found"})
            return

        try:
            request = self._read_json()
            replay_run_id = request.get("replayRunId")
            if not isinstance(replay_run_id, str) or not replay_run_id:
                self._send_json(400, {"status": "error", "message": "replayRunId is required"})
                return

            context = request.get("context") if isinstance(request.get("context"), dict) else {}
            experiments = _positive_int(context.get("experiments"), default=1)
            model = request.get("model") if isinstance(request.get("model"), str) else MODELS[0]
            db_path = _db_path_for_replay(replay_run_id)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            os.environ["RAINDROP_LOCAL_DEBUGGER"] = "http://localhost:5899/v1/"
            trace_input = {
                "experiments": experiments,
                "dbPath": str(db_path),
                "source": "workshop-replay",
                "systemPrompt": request.get("systemPrompt"),
                "userMessage": request.get("userMessage") or _last_user_message(request.get("messages")),
                "messageCount": len(request.get("messages") or []),
            }

            summaries = run_traced_dry_research_loop(
                REPO_ROOT,
                db_path,
                experiments,
                event_id=replay_run_id,
                user_id=str(context.get("userId") or "workshop-local"),
                convo_id=str(request.get("sourceRunId") or replay_run_id),
                source="workshop-replay",
                model=model,
                trace_input=trace_input,
            )
        except Exception as exc:
            self._send_json(
                500,
                {
                    "status": "error",
                    "message": str(exc),
                    "stack": traceback.format_exc(),
                },
            )
            return

        self._send_json(
            200,
            {
                "replayId": replay_run_id,
                "status": "done",
                "experiments": experiments,
                "summaryCount": len(summaries),
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or "0")
        body = self.rfile.read(length)
        if not body:
            return {}
        decoded = json.loads(body.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Replay request must be a JSON object")
        return decoded

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_artifact_file(self, query: str) -> None:
        path = _resolve_artifact_path(query)
        if path is None:
            self._send_json(400, {"ok": False, "error": "missing_path"})
            return
        if not path.exists() or not path.is_file():
            self._send_json(404, {"ok": False, "error": "artifact_not_found"})
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        byte_range = _parse_range_header(range_header, file_size) if range_header else None
        if range_header and byte_range is None:
            self.send_response(416)
            self.send_header("content-range", f"bytes */{file_size}")
            self.end_headers()
            return
        start, end = byte_range if byte_range is not None else (0, file_size - 1)
        length = max(0, end - start + 1)
        self.send_response(206 if byte_range is not None else 200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(length))
        self.send_header("accept-ranges", "bytes")
        if byte_range is not None:
            self.send_header("content-range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                self.wfile.write(chunk)

    def _send_artifact_video_page(self, query: str) -> None:
        path = _resolve_artifact_path(query)
        if path is None:
            self._send_json(400, {"ok": False, "error": "missing_path"})
            return
        if not path.exists() or not path.is_file():
            self._send_json(404, {"ok": False, "error": "artifact_not_found"})
            return
        rel = quote(str(path.relative_to(REPO_ROOT)))
        name = html.escape(path.name)
        src = f"/artifact-file?path={rel}"
        body = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>{name}</title>
    <style>
      html, body {{ margin: 0; height: 100%; background: #0f1115; color: #f4f4f5; font: 14px system-ui, sans-serif; }}
      main {{ display: flex; flex-direction: column; gap: 12px; height: 100%; box-sizing: border-box; padding: 16px; }}
      video {{ width: 100%; height: min(78vh, 720px); background: #000; border: 1px solid #2a2f3a; }}
      a {{ color: #93c5fd; }}
    </style>
  </head>
  <body>
    <main>
      <strong>{name}</strong>
      <video controls src="{src}"></video>
      <a href="{src}">Open raw MP4</a>
    </main>
  </body>
</html>"""
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _last_user_message(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            return json.dumps(content, sort_keys=True, default=str)
    return None


def _db_path_for_replay(replay_run_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", replay_run_id)[:80]
    return REPO_ROOT / "artifacts" / "replays" / f"{safe_id}.db"


def _resolve_artifact_path(query: str) -> Path | None:
    raw = (parse_qs(query).get("path") or [""])[0]
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(REPO_ROOT.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def _parse_range_header(header: str | None, file_size: int) -> tuple[int, int] | None:
    if not header or not header.startswith("bytes="):
        return None
    value = header.removeprefix("bytes=").split(",", 1)[0].strip()
    if "-" not in value:
        return None
    start_text, end_text = value.split("-", 1)
    try:
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        else:
            suffix = int(end_text)
            if suffix <= 0:
                return None
            start = max(0, file_size - suffix)
            end = file_size - 1
    except ValueError:
        return None
    if start < 0 or end < start or start >= file_size:
        return None
    return start, min(end, file_size - 1)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), ReplayHandler)
    print(f"Replay server listening on http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
