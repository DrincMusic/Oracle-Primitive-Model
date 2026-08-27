from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import time
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_running(pid: int) -> bool:
    if os.name == "nt":
        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        ctypes.windll.kernel32.CloseHandle(process)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(temporary, path)


def validate_manifest(args: argparse.Namespace) -> None:
    run_directory = args.run_root / args.run_id
    manifest_path = run_directory / "run-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"run manifest is missing for requested run ID: {args.run_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_revision = f"opm-source-sha256:{args.source_digest}"
    expected = {
        "run_id": args.run_id,
        "model_kind": args.condition,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "model_seed": args.seed,
        "code_revision": expected_revision,
    }
    observed = {
        "run_id": manifest.get("run_id"),
        "model_kind": manifest.get("configuration", {}).get("model_kind"),
        "learning_rate": manifest.get("configuration", {}).get("learning_rate"),
        "dropout": manifest.get("configuration", {}).get("dropout"),
        "model_seed": manifest.get("model_seed"),
        "code_revision": manifest.get("code_revision"),
    }
    if observed != expected:
        raise ValueError(f"monitor identity does not match run manifest: {observed!r}")


def event_progress(path: Path) -> tuple[int, int]:
    current_step = 0
    event_count = 0
    if not path.is_file():
        return current_step, event_count
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        current_step = int(event["step"])
        event_count += 1
    return current_step, event_count


def checkpoint_candidates(run_directory: Path) -> list[tuple[int, Path]]:
    candidates = []
    for path in (run_directory / "checkpoints").glob("step-*.pt"):
        match = re.fullmatch(r"step-(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return sorted(candidates)


def stable_checkpoint(
    run_directory: Path,
    observations: dict[Path, tuple[int, int]],
    *,
    terminal: bool,
) -> Path | None:
    stable: list[tuple[int, Path]] = []
    current: dict[Path, tuple[int, int]] = {}
    for step, path in checkpoint_candidates(run_directory):
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        current[path] = signature
        if terminal or observations.get(path) == signature:
            stable.append((step, path))
    observations.clear()
    observations.update(current)
    return stable[-1][1] if stable else None


def snapshot(
    args: argparse.Namespace, observations: dict[Path, tuple[int, int]] | None = None
) -> dict[str, object]:
    run_directory = args.run_root / args.run_id
    events_path = run_directory / "events.jsonl"
    current_step, event_count = event_progress(events_path)
    running = process_running(args.pid)
    completed = (run_directory / "summary.json").is_file()
    state = "COMPLETED" if completed else "RUNNING" if running else "INTERRUPTED"
    latest = stable_checkpoint(
        run_directory, observations if observations is not None else {}, terminal=state != "RUNNING"
    )
    return {
        "run_id": args.run_id,
        "condition": args.condition,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "seed": args.seed,
        },
        "current_step": current_step,
        "latest_checkpoint": str(latest.resolve()).replace("\\", "/") if latest else None,
        "checkpoint_sha256": sha256(latest) if latest else None,
        "process_id": args.pid,
        "process_state": state,
        "source_digest": args.source_digest,
        "last_event_offset": event_count,
        "monitor_accessed_sealed_labels": False,
        "monitor_performed_aggregate_evaluation": False,
        "monitor_performed_claim_decisions": False,
        "updated_unix_time": time.time(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    validate_manifest(args)
    last_checkpoint: str | None = None
    observations: dict[Path, tuple[int, int]] = {}
    while True:
        payload = snapshot(args, observations)
        checkpoint = payload["latest_checkpoint"]
        terminal = payload["process_state"] != "RUNNING"
        if checkpoint != last_checkpoint or terminal:
            atomic_json(args.output, payload)
            last_checkpoint = checkpoint if isinstance(checkpoint, str) else None
        if terminal:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
