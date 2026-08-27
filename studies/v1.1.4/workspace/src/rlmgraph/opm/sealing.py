from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .data import MaterializedExample
from .protocol import ProtocolState
from .splits import record_dict


@dataclass(frozen=True)
class SealManifest:
    input_path: str
    label_path: str
    input_sha256: str
    label_sha256: str
    row_count: int
    validation_only: bool


@dataclass(frozen=True)
class LockedEvaluationResult:
    accuracy: float
    correct: int
    count: int
    prediction_sha256: str
    label_sha256: str
    validation_harness: bool


def _jsonl(rows: Iterable[dict[str, object]]) -> bytes:
    return (
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n"
    ).encode("utf-8")


def seal_validation_labels(
    records: list[MaterializedExample], output_directory: Path, name: str
) -> SealManifest:
    output_directory.mkdir(parents=True, exist_ok=True)
    inputs: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for record in records:
        row = record_dict(record)
        label = int(row.pop("label"))
        inputs.append(row)
        labels.append({"example_id": record.example_id, "label": label})
    input_payload = _jsonl(inputs)
    label_payload = _jsonl(labels)
    input_path = output_directory / f"{name}.inputs.jsonl"
    label_path = output_directory / f"{name}.labels.sealed.jsonl"
    input_path.write_bytes(input_payload)
    label_path.write_bytes(label_payload)
    manifest = SealManifest(
        input_path=str(input_path),
        label_path=str(label_path),
        input_sha256=hashlib.sha256(input_payload).hexdigest(),
        label_sha256=hashlib.sha256(label_payload).hexdigest(),
        row_count=len(records),
        validation_only=True,
    )
    (output_directory / f"{name}.seal-manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def audit_label_access(audit_path: Path, *, actor: str, purpose: str, label_path: Path) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "actor": actor,
        "purpose": purpose,
        "label_path": str(label_path),
        "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
    }
    with audit_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")


def _audit_evaluator(
    audit_path: Path,
    *,
    actor: str,
    purpose: str,
    decision: str,
    label_sha256: str,
    detail: str,
) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "actor": actor,
        "purpose": purpose,
        "decision": decision,
        "label_sha256": label_sha256,
        "detail": detail,
    }
    with audit_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at line {line_number}") from error
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row {line_number} is not an object")
        rows.append(row)
    return rows


def evaluate_sealed_predictions(
    manifest_path: Path,
    prediction_path: Path,
    audit_path: Path,
    *,
    protocol: ProtocolState,
    actor: str,
    purpose: str,
    validation_harness: bool = False,
) -> LockedEvaluationResult:
    """Evaluate exact-key predictions without returning or embedding sealed labels."""
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = SealManifest(**manifest_payload)
    try:
        if validation_harness:
            protocol.require_validation()
            if not manifest.validation_only:
                raise PermissionError("validation harness cannot access a primary label seal")
        else:
            protocol.require_locked_evaluation()
            if manifest.validation_only:
                raise PermissionError("primary locked evaluator requires a primary seal")
    except PermissionError as error:
        _audit_evaluator(
            audit_path,
            actor=actor,
            purpose=purpose,
            decision="DENIED",
            label_sha256=manifest.label_sha256,
            detail=str(error),
        )
        raise

    input_path = Path(manifest.input_path)
    label_path = Path(manifest.label_path)
    input_payload = input_path.read_bytes()
    if hashlib.sha256(input_payload).hexdigest() != manifest.input_sha256:
        raise ValueError("sealed input fingerprint mismatch")

    prediction_payload = prediction_path.read_bytes()
    predictions = _read_jsonl(prediction_path)
    inputs = _read_jsonl(input_path)
    input_ids = [str(row.get("example_id")) for row in inputs]
    if len(input_ids) != manifest.row_count or len(set(input_ids)) != len(input_ids):
        raise ValueError("sealed inputs have invalid example keys")
    prediction_by_id: dict[str, int] = {}
    for row in predictions:
        if set(row) != {"example_id", "prediction"}:
            raise ValueError("prediction rows require exactly example_id and prediction")
        example_id = str(row["example_id"])
        prediction = row["prediction"]
        if example_id in prediction_by_id:
            raise ValueError(f"duplicate prediction key: {example_id}")
        if not isinstance(prediction, int) or isinstance(prediction, bool) or prediction not in (0, 1):
            raise ValueError(f"invalid prediction for {example_id}")
        prediction_by_id[example_id] = prediction
    if set(prediction_by_id) != set(input_ids):
        raise ValueError("prediction keys do not exactly match sealed inputs")

    # This is the only label-file read in the evaluator, after every authorization and input check.
    label_payload = label_path.read_bytes()
    if hashlib.sha256(label_payload).hexdigest() != manifest.label_sha256:
        raise ValueError("sealed label fingerprint mismatch")
    labels = _read_jsonl(label_path)
    label_by_id: dict[str, int] = {}
    for row in labels:
        if set(row) != {"example_id", "label"}:
            raise ValueError("sealed label rows require exactly example_id and label")
        example_id = str(row["example_id"])
        label = row["label"]
        if example_id in label_by_id:
            raise ValueError(f"duplicate sealed label key: {example_id}")
        if not isinstance(label, int) or isinstance(label, bool) or label not in (0, 1):
            raise ValueError(f"invalid sealed label for {example_id}")
        label_by_id[example_id] = label
    if set(label_by_id) != set(input_ids) or len(label_by_id) != manifest.row_count:
        raise ValueError("sealed label keys do not exactly match sealed inputs")

    correct = sum(prediction_by_id[key] == label_by_id[key] for key in input_ids)
    result = LockedEvaluationResult(
        accuracy=correct / manifest.row_count,
        correct=correct,
        count=manifest.row_count,
        prediction_sha256=hashlib.sha256(prediction_payload).hexdigest(),
        label_sha256=manifest.label_sha256,
        validation_harness=validation_harness,
    )
    _audit_evaluator(
        audit_path,
        actor=actor,
        purpose=purpose,
        decision="ALLOWED_VALIDATION" if validation_harness else "ALLOWED_LOCKED",
        label_sha256=manifest.label_sha256,
        detail=f"count={result.count};correct={result.correct}",
    )
    return result
