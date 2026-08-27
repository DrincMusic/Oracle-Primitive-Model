import argparse
import json
from pathlib import Path

import pytest

from scripts.opm_live_state_monitor import (
    atomic_json,
    checkpoint_candidates,
    event_progress,
    snapshot,
    stable_checkpoint,
    validate_manifest,
)


def _args(tmp_path: Path, **changes):
    values = {
        "run_root": tmp_path,
        "run_id": "run-1",
        "condition": "OPM_SHARED",
        "learning_rate": 0.0001,
        "dropout": 0.0,
        "seed": 1101,
        "pid": 99999999,
        "source_digest": "abc",
    }
    values.update(changes)
    return argparse.Namespace(**values)


def _manifest(tmp_path: Path) -> Path:
    run = tmp_path / "run-1"
    run.mkdir()
    (run / "run-manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "configuration": {
                    "model_kind": "OPM_SHARED",
                    "learning_rate": 0.0001,
                    "dropout": 0.0,
                },
                "model_seed": 1101,
                "code_revision": "opm-source-sha256:abc",
            }
        ),
        encoding="utf-8",
    )
    return run


def test_partial_final_event_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"step":1}\n{"step":', encoding="utf-8")
    assert event_progress(path) == (1, 1)


def test_checkpoint_is_numeric_and_requires_two_live_polls(tmp_path: Path) -> None:
    run = _manifest(tmp_path)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "step-999.pt").write_bytes(b"old")
    newest = checkpoints / "step-10000.pt"
    newest.write_bytes(b"new")
    assert [step for step, _ in checkpoint_candidates(run)] == [999, 10000]
    observations = {}
    assert stable_checkpoint(run, observations, terminal=False) is None
    assert stable_checkpoint(run, observations, terminal=False) == newest
    newest.write_bytes(b"incomplete-change")
    assert stable_checkpoint(run, observations, terminal=False).name == "step-999.pt"


def test_interruption_completion_and_atomic_replace(tmp_path: Path) -> None:
    run = _manifest(tmp_path)
    (run / "events.jsonl").write_text('{"step":2}\n', encoding="utf-8")
    interrupted = snapshot(_args(tmp_path), {})
    assert interrupted["process_state"] == "INTERRUPTED"
    (run / "summary.json").write_text("{}", encoding="utf-8")
    completed = snapshot(_args(tmp_path), {})
    assert completed["process_state"] == "COMPLETED"
    output = tmp_path / "live.json"
    atomic_json(output, completed)
    assert json.loads(output.read_text(encoding="utf-8"))["process_state"] == "COMPLETED"
    assert not output.with_suffix(".json.tmp").exists()


def test_manifest_identity_fails_closed(tmp_path: Path) -> None:
    _manifest(tmp_path)
    validate_manifest(_args(tmp_path))
    with pytest.raises(ValueError, match="does not match"):
        validate_manifest(_args(tmp_path, source_digest="wrong"))
    with pytest.raises(ValueError, match="manifest"):
        validate_manifest(_args(tmp_path, run_id="wrong-run"))
