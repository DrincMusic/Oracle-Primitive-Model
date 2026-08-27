from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .generation import Operation
from .rendering import Domain
from .splits import HOLDOUTS, SplitName, canonical_expected_rows


@dataclass(frozen=True)
class DatasetAudit:
    passed: bool
    total_rows: int
    split_rows: dict[str, int]
    unique_worlds: int
    unique_cross_split_latents: int
    test_labels_separated: bool
    sealed_label_content_accessed: bool
    checks: tuple[str, ...]


def audit_canonical_directory(directory: Path) -> DatasetAudit:
    world_owner: dict[int, SplitName] = {}
    latent_owner: dict[str, SplitName] = {}
    split_rows: dict[str, int] = {}
    checks: list[str] = []
    for split in SplitName:
        data_path = directory / f"{split.value}.canonical.jsonl"
        manifest_path = directory / f"{split.value}.canonical.manifest.json"
        payload = data_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
            raise ValueError(f"dataset hash mismatch: {split.value}")
        rows = [json.loads(line) for line in payload.splitlines()]
        expected = canonical_expected_rows(split)
        if len(rows) != expected or manifest["row_count"] != expected:
            raise ValueError(f"row-count mismatch: {split.value}")
        split_rows[split.value] = len(rows)
        test_split = split not in (SplitName.TRAIN, SplitName.VALIDATION)
        input_ids: set[str] = set()
        for row in rows:
            if test_split and "label" in row:
                raise ValueError(f"test label exposed in inputs: {split.value}")
            world_id = int(row["world_id"])
            owner = world_owner.setdefault(world_id, split)
            if owner != split:
                raise ValueError(f"world contamination: {world_id}")
            latent = json.dumps(
                [row["facts"], row["query"]], sort_keys=True, separators=(",", ":")
            )
            latent_owner.setdefault(latent, split)
            if latent_owner[latent] != split:
                raise ValueError(f"latent contamination: {row['example_id']}")
            operation = Operation(int(row["operation_ids"][0]))
            domain = Domain[row["domain"]]
            if split == SplitName.TRAIN:
                if int(row["renderer_variant"]) == 2:
                    raise ValueError("renderer variant 2 in training")
                if operation == HOLDOUTS[domain]:
                    raise ValueError("held-out operation in training")
            input_ids.add(row["example_id"])
        if test_split:
            label_path = directory / "sealed-labels" / f"{split.value}.labels.jsonl"
            label_payload = label_path.read_bytes()
            if hashlib.sha256(label_payload).hexdigest() != manifest["labels_sha256"]:
                raise ValueError(f"label hash mismatch: {split.value}")
            label_rows = [json.loads(line) for line in label_payload.splitlines()]
            label_ids = {row["example_id"] for row in label_rows}
            if label_ids != input_ids or len(label_rows) != len(rows):
                raise ValueError(f"input/label key mismatch: {split.value}")
        checks.append(f"{split.value}:hash,count,policy")
    return DatasetAudit(
        passed=True,
        total_rows=sum(split_rows.values()),
        split_rows=split_rows,
        unique_worlds=len(world_owner),
        unique_cross_split_latents=len(latent_owner),
        test_labels_separated=True,
        sealed_label_content_accessed=True,
        checks=tuple(checks),
    )


def audit_v113_inputs_without_labels(directory: Path) -> DatasetAudit:
    """Audit v1.1.3 inputs and seal metadata without opening primary label files."""
    world_owner: dict[int, SplitName] = {}
    latent_owner: dict[str, SplitName] = {}
    split_rows: dict[str, int] = {}
    checks: list[str] = []
    for split in SplitName:
        data_path = directory / f"{split.value}.v1.1.3.canonical.jsonl"
        manifest_path = directory / f"{split.value}.v1.1.3.canonical.manifest.json"
        payload = data_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
            raise ValueError(f"dataset hash mismatch: {split.value}")
        rows = [json.loads(line) for line in payload.splitlines()]
        expected = canonical_expected_rows(split)
        if len(rows) != expected or manifest["row_count"] != expected:
            raise ValueError(f"row-count mismatch: {split.value}")
        split_rows[split.value] = len(rows)
        test_split = split not in (SplitName.TRAIN, SplitName.VALIDATION)
        if test_split:
            if not manifest.get("labels_separated") or not manifest.get("labels_sha256"):
                raise ValueError(f"test label seal metadata missing: {split.value}")
            label_path = directory / "sealed-labels" / f"{split.value}.v1.1.3.labels.jsonl"
            if not label_path.is_file():
                raise ValueError(f"sealed label artifact missing: {split.value}")
        for row in rows:
            if test_split and "label" in row:
                raise ValueError(f"test label exposed in inputs: {split.value}")
            if not test_split and "label" not in row:
                raise ValueError(f"training label missing: {split.value}")
            world_id = int(row["world_id"])
            owner = world_owner.setdefault(world_id, split)
            if owner != split:
                raise ValueError(f"world contamination: {world_id}")
            latent = json.dumps(
                [row["facts"], row["query"]], sort_keys=True, separators=(",", ":")
            )
            previous = latent_owner.setdefault(latent, split)
            if previous != split:
                raise ValueError(f"latent contamination: {row['example_id']}")
            operation = Operation(int(row["operation_ids"][0]))
            domain = Domain[row["domain"]]
            if split == SplitName.TRAIN:
                if int(row["renderer_variant"]) == 2:
                    raise ValueError("renderer variant 2 in training")
                if operation == HOLDOUTS[domain]:
                    raise ValueError("held-out operation in training")
        checks.append(f"{split.value}:input-hash,count,policy,contamination")
    return DatasetAudit(
        passed=True,
        total_rows=sum(split_rows.values()),
        split_rows=split_rows,
        unique_worlds=len(world_owner),
        unique_cross_split_latents=len(latent_owner),
        test_labels_separated=True,
        sealed_label_content_accessed=False,
        checks=tuple(checks),
    )


def write_dataset_audit(audit: DatasetAudit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(audit), indent=2, sort_keys=True) + "\n", encoding="utf-8")
