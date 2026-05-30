"""Watch Modal phase-1 jobs and surface failures quickly.

The guardian is intentionally conservative: it polls app status and log tails,
records JSONL events, and can relaunch an explicit phase-1 spec through the
deployed app when requested. It never edits code or cancels jobs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modal_runner.deployed import DEFAULT_APP_NAME, submit_phase1_specs_to_deployed


ERROR_PATTERNS = (
    "Traceback (most recent call last)",
    "subprocess.CalledProcessError",
    "RuntimeError:",
    "FileNotFoundError:",
    "ModuleNotFoundError:",
    "CUDA out of memory",
    "No space left on device",
    "No checkpoints found",
    "Could not locate isaaclab.sh",
    "HTTP Error",
    "InputCancellation",
    "CancelledError",
    "There was an error running python",
    "unrecognized arguments:",
    "Function failed",
    "Task failed",
    "Container failed",
)


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LogCheck:
    app_id: str
    command: CommandResult
    error_hits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "command": self.command.to_dict(),
            "error_hits": self.error_hits,
        }


@dataclass(frozen=True)
class GuardianEvent:
    timestamp: str
    app_list: CommandResult
    log_checks: list[LogCheck]
    relaunch_call_ids: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(check.error_hits or check.command.returncode != 0 for check in self.log_checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "app_list": self.app_list.to_dict(),
            "log_checks": [check.to_dict() for check in self.log_checks],
            "has_errors": self.has_errors,
            "relaunch_call_ids": self.relaunch_call_ids,
        }


def scan_for_modal_errors(text: str) -> list[str]:
    """Return matching error lines from a Modal log tail."""

    hits: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(pattern.lower() in lowered for pattern in ERROR_PATTERNS):
            cleaned = line.strip()
            if cleaned and cleaned not in hits:
                hits.append(cleaned)
    return hits


def poll_modal_once(*, app_ids: list[str], tail: int = 80, timeout_seconds: int = 45) -> GuardianEvent:
    app_list = run_command(["modal", "app", "list"], timeout_seconds=timeout_seconds)
    checks: list[LogCheck] = []
    for app_id in app_ids:
        result = run_command(["modal", "app", "logs", app_id, "--tail", str(tail)], timeout_seconds=timeout_seconds)
        checks.append(LogCheck(app_id=app_id, command=result, error_hits=scan_for_modal_errors(result.stdout)))
    return GuardianEvent(
        timestamp=datetime.now(UTC).isoformat(),
        app_list=app_list,
        log_checks=checks,
    )


def run_command(args: list[str], *, timeout_seconds: int) -> CommandResult:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except FileNotFoundError as exc:
        return CommandResult(args=args, returncode=127, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            args=args,
            returncode=124,
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr if isinstance(exc.stderr, str) else f"timed out after {timeout_seconds}s",
        )
    return CommandResult(args=args, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def append_event(path: Path, event: GuardianEvent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


def load_phase1_specs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, list) else [payload]


def relaunch_specs(path: Path, *, app_name: str = DEFAULT_APP_NAME) -> list[str]:
    return submit_phase1_specs_to_deployed(load_phase1_specs(path), app_name=app_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-id", action="append", default=[])
    parser.add_argument("--tail", type=int, default=80)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--iterations", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument("--output", default="artifacts/modal_guardian/events.jsonl")
    parser.add_argument("--phase1-spec", default="")
    parser.add_argument("--relaunch-on-error", action="store_true")
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    args = parser.parse_args()

    if not args.app_id:
        raise SystemExit("Provide at least one --app-id to watch")

    output = Path(args.output)
    iteration = 0
    while True:
        iteration += 1
        event = poll_modal_once(app_ids=args.app_id, tail=args.tail)
        relaunch_call_ids: list[str] = []
        if event.has_errors and args.relaunch_on_error and args.phase1_spec:
            relaunch_call_ids = relaunch_specs(Path(args.phase1_spec), app_name=args.app_name)
            event = GuardianEvent(
                timestamp=event.timestamp,
                app_list=event.app_list,
                log_checks=event.log_checks,
                relaunch_call_ids=relaunch_call_ids,
            )
        append_event(output, event)
        print(json.dumps(_summarize_event(event), indent=2, sort_keys=True), flush=True)

        if args.iterations and iteration >= args.iterations:
            break
        time.sleep(args.interval_seconds)


def _summarize_event(event: GuardianEvent) -> dict[str, Any]:
    return {
        "timestamp": event.timestamp,
        "has_errors": event.has_errors,
        "apps_checked": [check.app_id for check in event.log_checks],
        "error_hits": {
            check.app_id: check.error_hits or ([check.command.stderr.strip()] if check.command.returncode else [])
            for check in event.log_checks
            if check.error_hits or check.command.returncode
        },
        "relaunch_call_ids": event.relaunch_call_ids,
    }


if __name__ == "__main__":
    main()
