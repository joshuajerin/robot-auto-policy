"""Raindrop Workshop tracing helpers for RoboGenesis runs."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_SIM_EVENT_NAME = "robogenesis-sim-run"
DEFAULT_USER_ID = "robogenesis-local"
DEFAULT_LOCAL_WORKSHOP = "http://localhost:5899/v1/"
DEFAULT_REPLAY_PORT = 61020
TRACE_MANIFEST_NAME = "raindrop_trace.json"

_RAINDROP_SDK: Any | None = None
_RAINDROP_SDK_INITIALIZED = False
_WORKSHOP_PROBE_CACHE: dict[str, bool] = {}


def now_ms() -> int:
    return int(time.time() * 1000)


def artifact_file_url(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
    port: int = DEFAULT_REPLAY_PORT,
) -> str:
    return f"http://127.0.0.1:{port}/artifact-file?path={_quote_artifact_path(path, repo_root=repo_root)}"


def artifact_video_page_url(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
    port: int = DEFAULT_REPLAY_PORT,
) -> str:
    return f"http://127.0.0.1:{port}/artifact-video?path={_quote_artifact_path(path, repo_root=repo_root)}"


def build_video_markdown(
    video_paths: list[str],
    *,
    repo_root: str | Path | None = None,
    port: int = DEFAULT_REPLAY_PORT,
) -> str:
    lines: list[str] = []
    for index, path in enumerate(video_paths, start=1):
        name = Path(path).name or f"rollout-{index}.mp4"
        page_url = artifact_video_page_url(path, repo_root=repo_root, port=port)
        file_url = artifact_file_url(path, repo_root=repo_root, port=port)
        lines.append(f"- [Open {name}]({page_url})")
        lines.append(f'<video controls src="{file_url}" style="width:100%;max-width:900px"></video>')
    return "\n".join(lines)


def build_video_artifacts(
    video_paths: list[str],
    *,
    repo_root: str | Path | None = None,
    port: int = DEFAULT_REPLAY_PORT,
) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for index, path in enumerate(video_paths, start=1):
        name = Path(path).name or f"rollout-{index}.mp4"
        artifacts.append(
            {
                "label": name,
                "path": str(path),
                "video_page_url": artifact_video_page_url(path, repo_root=repo_root, port=port),
                "file_url": artifact_file_url(path, repo_root=repo_root, port=port),
            }
        )
    return artifacts


@dataclass
class RaindropRun:
    event_name: str
    event_id: str
    user_id: str = DEFAULT_USER_ID
    convo_id: str | None = None
    input_payload: Any | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    local_workshop_url: str = DEFAULT_LOCAL_WORKSHOP
    enabled: bool = False
    local_enabled: bool = False
    interaction: Any | None = None
    trace_id: str = ""
    root_span_id: str = ""
    started_ms: int = field(default_factory=now_ms)
    ended: bool = False

    @classmethod
    def start(
        cls,
        *,
        event_name: str = DEFAULT_SIM_EVENT_NAME,
        event_id: str | None = None,
        user_id: str = DEFAULT_USER_ID,
        convo_id: str | None = None,
        input_payload: Any | None = None,
        properties: dict[str, Any] | None = None,
        local_workshop_url: str | None = None,
    ) -> "RaindropRun":
        event_id = event_id or f"{event_name}-{secrets.token_hex(8)}"
        local_url = normalize_local_workshop_url(local_workshop_url)
        local_enabled = workshop_available(local_url)
        enabled = local_enabled or bool(os.environ.get("RAINDROP_WRITE_KEY"))
        run = cls(
            event_name=event_name,
            event_id=event_id,
            user_id=user_id,
            convo_id=convo_id,
            input_payload=input_payload,
            properties=properties or {},
            local_workshop_url=local_url,
            enabled=enabled,
            local_enabled=local_enabled,
            trace_id=_trace_id_for_event(event_id),
            root_span_id=_span_id(),
            started_ms=now_ms(),
        )
        if not enabled:
            return run

        sdk = init_raindrop_sdk(local_url)
        if sdk is not None:
            try:
                run.interaction = sdk.begin(
                    user_id=user_id,
                    event=event_name,
                    event_id=event_id,
                    input=_json_text(input_payload),
                    convo_id=convo_id,
                    properties=run.properties,
                )
            except Exception:
                run.interaction = None
        return run

    def record_task(
        self,
        name: str,
        *,
        input_payload: Any | None = None,
        output_payload: Any | None = None,
        error: BaseException | str | None = None,
        properties: dict[str, Any] | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        start = start_ms or now_ms()
        end = end_ms or now_ms()
        if end < start:
            end = start
        attrs = self._base_attrs(properties)
        attrs.extend(
            [
                _str_attr("raindrop.span.kind", "tool_call"),
                _str_attr("traceloop.span.kind", "tool"),
                _str_attr("traceloop.entity.name", name),
                _int_attr("traceloop.entity.duration_ms", end - start),
            ]
        )
        if input_payload is not None:
            attrs.append(_str_attr("traceloop.entity.input", _json_text(input_payload)))
        if output_payload is not None or error is not None:
            attrs.append(_str_attr("traceloop.entity.output", _json_text(output_payload if error is None else {"error": str(error)})))
        self._post_span(
            span_id=_span_id(),
            parent_span_id=self.root_span_id,
            name=name,
            attributes=attrs,
            start_ms=start,
            end_ms=end,
            error=error,
        )

    def finish(
        self,
        *,
        output_payload: Any | None = None,
        status: str = "done",
        error: BaseException | str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        if self.ended:
            return
        self.ended = True
        merged = {**self.properties, **(properties or {}), "status": status}
        output_text = _json_text(output_payload)

        if self.enabled:
            if self.interaction is not None:
                try:
                    self.interaction.finish(output=output_text, properties=merged)
                except Exception:
                    pass
            attrs = self._base_attrs(merged)
            attrs.extend(
                [
                    _str_attr("raindrop.span.kind", "agent_root"),
                    _str_attr("traceloop.span.kind", "task"),
                    _str_attr("traceloop.entity.name", self.event_name),
                    _int_attr("traceloop.entity.duration_ms", now_ms() - self.started_ms),
                ]
            )
            if self.input_payload is not None:
                attrs.append(_str_attr("traceloop.entity.input", _json_text(self.input_payload)))
            if output_payload is not None or error is not None:
                attrs.append(_str_attr("traceloop.entity.output", output_text if error is None else _json_text({"error": str(error)})))
            self._post_span(
                span_id=self.root_span_id,
                parent_span_id=None,
                name=self.event_name,
                attributes=attrs,
                start_ms=self.started_ms,
                end_ms=now_ms(),
                error=error,
            )
            flush_raindrop_sdk()

    def _base_attrs(self, properties: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        attrs = [
            _str_attr("traceloop.association.properties.event_id", self.event_id),
            _str_attr("traceloop.association.properties.event_name", self.event_name),
            _str_attr("traceloop.association.properties.user_id", self.user_id),
        ]
        if self.convo_id:
            attrs.append(_str_attr("traceloop.association.properties.convo_id", self.convo_id))
        for key, value in (properties or {}).items():
            if value is None:
                continue
            attrs.append(_association_attr(key, value))
        return attrs

    def _post_span(
        self,
        *,
        span_id: str,
        parent_span_id: str | None,
        name: str,
        attributes: list[dict[str, Any]],
        start_ms: int,
        end_ms: int,
        error: BaseException | str | None = None,
    ) -> None:
        if not self.local_enabled:
            return
        span: dict[str, Any] = {
            "traceId": self.trace_id,
            "spanId": span_id,
            "name": name,
            "startTimeUnixNano": str(start_ms * 1_000_000),
            "endTimeUnixNano": str(end_ms * 1_000_000),
            "attributes": attributes,
            "status": {"code": 2, "message": str(error)} if error is not None else {"code": 1},
        }
        if parent_span_id:
            span["parentSpanId"] = parent_span_id
        payload = {
            "resourceSpans": [
                {
                    "resource": {"attributes": [_str_attr("service.name", "robogenesis")]},
                    "scopeSpans": [
                        {
                            "scope": {"name": "robogenesis", "version": "0.1.0"},
                            "spans": [span],
                        }
                    ],
                }
            ]
        }
        post_workshop_json(self.local_workshop_url, "traces", payload)


def publish_artifact_run(
    artifact_dir: str | Path,
    *,
    ingest_summary: dict[str, Any] | None = None,
    accepted: bool | None = None,
    review_reasons: list[str] | None = None,
    db_path: str | Path | None = None,
    event_id: str | None = None,
    user_id: str = DEFAULT_USER_ID,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    manifest = _load_json_if_exists(artifact_root / TRACE_MANIFEST_NAME) or {}
    ingest_summary = ingest_summary or {}
    experiment_id = str(
        ingest_summary.get("experiment_id")
        or manifest.get("experiment_id")
        or artifact_root.name
    )
    event_id = str(event_id or manifest.get("event_id") or experiment_id)
    event_name = str(manifest.get("event_name") or DEFAULT_SIM_EVENT_NAME)
    videos = _video_paths(artifact_root, manifest, ingest_summary)
    video_artifacts = build_video_artifacts(videos, repo_root=repo_root)
    video_pages = [artifact["video_page_url"] for artifact in video_artifacts]
    properties = {
        "experiment_id": experiment_id,
        "artifact_dir": str(artifact_root),
        "accepted": accepted,
        "review_reasons": review_reasons or [],
        "dbPath": str(db_path) if db_path is not None else None,
        "video_count": len(videos),
        "primary_video_path": videos[0] if videos else None,
        "primary_video_page": video_pages[0] if video_pages else None,
        "video_pages": video_pages,
        "video_artifacts": video_artifacts,
        "source": "artifact-sync",
    }
    run = RaindropRun.start(
        event_name=event_name,
        event_id=event_id,
        user_id=user_id,
        convo_id=experiment_id,
        input_payload={
            "experiment_id": experiment_id,
            "artifact_dir": str(artifact_root),
            "manifest": manifest,
        },
        properties=properties,
    )
    for task in _manifest_tasks(manifest):
        task_properties = {
            "experiment_id": experiment_id,
            "task_status": task.get("status"),
            **(task.get("properties") if isinstance(task.get("properties"), dict) else {}),
        }
        if _task_should_show_videos(task) and video_artifacts:
            task_properties.update(
                {
                    "video_count": len(video_artifacts),
                    "primary_video_page": video_artifacts[0]["video_page_url"],
                }
            )
        run.record_task(
            str(task.get("name") or "modal_task"),
            input_payload=task.get("input"),
            output_payload=_task_output_with_videos(task, task.get("output"), video_artifacts),
            error=task.get("error") if task.get("status") == "error" else None,
            properties=task_properties,
            start_ms=_int_or_none(task.get("start_ms")),
            end_ms=_int_or_none(task.get("end_ms")),
        )
    run.record_task(
        "sync_and_ingest_artifacts",
        input_payload={"artifact_dir": str(artifact_root), "db_path": str(db_path) if db_path is not None else None},
        output_payload={
            "ingest_summary": ingest_summary,
            "accepted": accepted,
            "review_reasons": review_reasons or [],
            "video_pages": video_pages,
            "render_videos": video_artifacts,
        },
        properties={"experiment_id": experiment_id},
    )
    output = _artifact_run_output(
        experiment_id=experiment_id,
        ingest_summary=ingest_summary,
        accepted=accepted,
        review_reasons=review_reasons or [],
        videos=videos,
        repo_root=repo_root,
    )
    run.finish(output_payload=output, status="accepted" if accepted else "reviewed", properties=properties)
    return {
        "enabled": run.enabled,
        "local_enabled": run.local_enabled,
        "event_name": event_name,
        "event_id": event_id,
        "trace_id": run.trace_id,
        "task_count": len(_manifest_tasks(manifest)) + 1,
        "video_pages": video_pages,
    }


def normalize_local_workshop_url(raw: str | None = None) -> str:
    raw = raw or os.environ.get("RAINDROP_LOCAL_DEBUGGER") or os.environ.get("RAINDROP_WORKSHOP") or DEFAULT_LOCAL_WORKSHOP
    raw = raw.strip()
    if raw.lower() in {"1", "true", "yes", "on"}:
        return DEFAULT_LOCAL_WORKSHOP
    if raw.lower() in {"0", "false", "no", "off"}:
        return ""
    if not raw:
        return DEFAULT_LOCAL_WORKSHOP
    if not raw.startswith(("http://", "https://")):
        return DEFAULT_LOCAL_WORKSHOP
    parsed = urllib.parse.urlparse(raw)
    if parsed.path.rstrip("/") == "/v1":
        return raw if raw.endswith("/") else f"{raw}/"
    base = f"{parsed.scheme}://{parsed.netloc}"
    return f"{base}/v1/"


def workshop_available(local_workshop_url: str | None = None) -> bool:
    local_url = normalize_local_workshop_url(local_workshop_url)
    if not local_url:
        return False
    if local_url in _WORKSHOP_PROBE_CACHE:
        return _WORKSHOP_PROBE_CACHE[local_url]
    probe_url = local_url
    try:
        request = urllib.request.Request(probe_url, method="HEAD")
        with urllib.request.urlopen(request, timeout=0.15):
            available = True
    except urllib.error.HTTPError as exc:
        available = exc.code < 500
    except Exception:
        available = False
    _WORKSHOP_PROBE_CACHE[local_url] = available
    return available


def init_raindrop_sdk(local_workshop_url: str | None = None) -> Any | None:
    global _RAINDROP_SDK, _RAINDROP_SDK_INITIALIZED
    if _RAINDROP_SDK_INITIALIZED:
        return _RAINDROP_SDK
    _RAINDROP_SDK_INITIALIZED = True
    try:
        import raindrop.analytics as raindrop
    except ImportError:
        return None
    try:
        raindrop.init(
            api_key=os.environ.get("RAINDROP_WRITE_KEY") or None,
            local_workshop_url=normalize_local_workshop_url(local_workshop_url),
            tracing_enabled=False,
            auto_instrument=False,
        )
    except Exception:
        return None
    _RAINDROP_SDK = raindrop
    return _RAINDROP_SDK


def flush_raindrop_sdk() -> None:
    if _RAINDROP_SDK is None:
        return
    try:
        _RAINDROP_SDK.flush()
    except Exception:
        pass


def post_workshop_json(local_workshop_url: str, path: str, payload: Any) -> bool:
    if not local_workshop_url:
        return False
    url = f"{local_workshop_url.rstrip('/')}/{path.lstrip('/')}"
    data = json.dumps(payload, default=str).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=0.25):
            return True
    except Exception:
        return False


def _artifact_run_output(
    *,
    experiment_id: str,
    ingest_summary: dict[str, Any],
    accepted: bool | None,
    review_reasons: list[str],
    videos: list[str],
    repo_root: Path,
) -> str:
    lines = [
        f"# RoboGenesis sim run `{experiment_id}`",
        "",
        f"- status: {'accepted' if accepted else 'reviewed'}",
        f"- score: {ingest_summary.get('score', 'unknown')}",
        f"- primary failure: {ingest_summary.get('primary_failure', 'unknown')}",
    ]
    if review_reasons:
        lines.append(f"- review reasons: {', '.join(review_reasons)}")
    if videos:
        lines.extend(["", "## Render video", "", build_video_markdown(videos, repo_root=repo_root)])
    else:
        lines.extend(["", "No rollout video was found in the synced artifacts."])
    return "\n".join(lines)


def _manifest_tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [task for task in tasks if isinstance(task, dict)]


def _task_should_show_videos(task: dict[str, Any]) -> bool:
    name = str(task.get("name") or "").lower()
    if "render" in name or "video" in name:
        return True
    output = task.get("output")
    if isinstance(output, dict):
        return bool(
            output.get("video_paths")
            or output.get("actual_video_paths")
            or output.get("all_video_paths")
            or output.get("primary_video_path")
            or output.get("primary_actual_video_path")
        )
    return False


def _task_output_with_videos(
    task: dict[str, Any],
    output: Any,
    video_artifacts: list[dict[str, str]],
) -> Any:
    if not video_artifacts or not _task_should_show_videos(task):
        return output
    if isinstance(output, dict):
        return {**output, "render_videos": video_artifacts}
    return {"output": output, "render_videos": video_artifacts}


def _video_paths(artifact_root: Path, manifest: dict[str, Any], ingest_summary: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("rollout_video_paths", "video_paths", "actual_video_paths", "all_video_paths"):
        value = ingest_summary.get(key)
        if isinstance(value, list):
            values.extend(value)
        value = manifest.get(key)
        if isinstance(value, list):
            values.extend(value)
    for key in ("rollout_video_path", "primary_video_path", "primary_actual_video_path"):
        value = ingest_summary.get(key) or manifest.get(key)
        if value:
            values.append(value)
    if not values and artifact_root.exists():
        values.extend(str(path) for path in sorted(artifact_root.rglob("*.mp4")))

    resolved: list[str] = []
    for value in values:
        path = _localize_artifact_path(artifact_root, value)
        text = str(path)
        if text not in resolved:
            resolved.append(text)
    return resolved


def _localize_artifact_path(artifact_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.exists():
        return path.resolve()

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.extend(artifact_root / suffix for suffix in _artifact_path_suffixes(path, artifact_root.name))
        candidates.append(artifact_root / path.name)
    else:
        candidates.extend([artifact_root / path, path])
        candidates.extend(artifact_root / suffix for suffix in _artifact_path_suffixes(path, artifact_root.name))
        candidates.append(artifact_root / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def _artifact_path_suffixes(path: Path, experiment_id: str) -> list[Path]:
    suffixes: list[Path] = []
    parts = path.parts
    for index, part in enumerate(parts):
        if part == experiment_id and index + 1 < len(parts):
            suffixes.append(Path(*parts[index + 1 :]))
        if part == "experiments" and index + 2 < len(parts):
            suffixes.append(Path(*parts[index + 2 :]))
    return _dedupe_paths(suffixes)


def _quote_artifact_path(path: str | Path, *, repo_root: str | Path | None) -> str:
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo / candidate
    try:
        value = str(candidate.resolve().relative_to(repo.resolve()))
    except (OSError, ValueError):
        value = str(candidate)
    return urllib.parse.quote(value)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _trace_id_for_event(event_id: str) -> str:
    value = hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:32]
    return value if value != "0" * 32 else "1" + value[1:]


def _span_id() -> str:
    return secrets.token_hex(8)


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_attr(key: str, value: Any) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": str(value)}}


def _int_attr(key: str, value: int) -> dict[str, Any]:
    return {"key": key, "value": {"intValue": str(int(value))}}


def _bool_attr(key: str, value: bool) -> dict[str, Any]:
    return {"key": key, "value": {"boolValue": bool(value)}}


def _double_attr(key: str, value: float) -> dict[str, Any]:
    return {"key": key, "value": {"doubleValue": float(value)}}


def _association_attr(key: str, value: Any) -> dict[str, Any]:
    attr_key = f"traceloop.association.properties.{key}"
    if isinstance(value, bool):
        return _bool_attr(attr_key, value)
    if isinstance(value, int) and not isinstance(value, bool):
        return _int_attr(attr_key, value)
    if isinstance(value, float):
        return _double_attr(attr_key, value)
    if isinstance(value, (dict, list, tuple)):
        return _str_attr(attr_key, _json_text(value))
    return _str_attr(attr_key, value)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped
