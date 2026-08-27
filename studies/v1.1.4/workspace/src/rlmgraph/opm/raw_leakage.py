from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .probes import RawLeakageProbeResult, categorical_leakage_probe, raw_probe_family_passes

_ENTITY_ALIAS = re.compile(r"\b(?:e|s|o|r|v|c)\d{2}\b")


@dataclass(frozen=True)
class RawLeakageReport:
    specification_version: str
    train_rows: int
    validation_rows: int
    feature_schemas: dict[str, tuple[str, ...]]
    results: tuple[RawLeakageProbeResult, ...]
    holm_family_passes: bool
    protocol_freeze_gate_passes: bool
    argument_source: str = "surface_alias"


def _features(row: dict[str, object], argument_source: str = "surface_alias") -> dict[str, list[str]]:
    step_mask = list(row["step_mask"])
    operations = list(row["operation_ids"])
    evidence = list(row["evidence_indices"])
    if argument_source == "serialized_binding":
        aliases = [str(value) for value in row["argument_entity_ids"]]
    elif argument_source == "surface_alias":
        aliases = _ENTITY_ALIAS.findall(str(row["rendered_query"]))
        if len(aliases) != 2:
            raise ValueError(f"expected two randomized query aliases: {row['example_id']}")
    else:
        raise ValueError(f"unknown argument source: {argument_source}")
    return {
        "operation_and_step": [
            str(operations[0]),
            str(operations[1]) if step_mask[1] else "ABSENT",
            "2" if step_mask[1] else "1",
        ],
        "evidence_positions": [
            str(evidence[0]),
            str(evidence[1]) if step_mask[1] else "ABSENT",
        ],
        "renamed_argument_ids": aliases,
    }


def _load(
    path: Path, argument_source: str = "surface_alias"
) -> tuple[dict[str, list[list[str]]], np.ndarray]:
    channels: dict[str, list[list[str]]] = {
        "operation_and_step": [],
        "evidence_positions": [],
        "renamed_argument_ids": [],
    }
    labels: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "label" not in row:
                raise ValueError(f"probe input has no labels: {path}")
            labels.append(int(row["label"]))
            for channel, values in _features(row, argument_source).items():
                channels[channel].append(values)
    return channels, np.asarray(labels, dtype=np.int64)


def run_raw_leakage_gates(
    train_path: Path, validation_path: Path, *, argument_source: str = "surface_alias"
) -> RawLeakageReport:
    train, train_labels = _load(train_path, argument_source)
    validation, validation_labels = _load(validation_path, argument_source)
    results = tuple(
        categorical_leakage_probe(
            train[channel],
            train_labels,
            validation[channel],
            validation_labels,
            channel=channel,
        )
        for channel in train
    )
    passes = raw_probe_family_passes(list(results))
    return RawLeakageReport(
        specification_version="1.0.0",
        train_rows=len(train_labels),
        validation_rows=len(validation_labels),
        feature_schemas={
            "operation_and_step": ("operation_1", "operation_2_or_absent", "step_count"),
            "evidence_positions": ("evidence_position_1", "evidence_position_2_or_absent"),
            "renamed_argument_ids": (
                ("serialized_binding_argument_1", "serialized_binding_argument_2")
                if argument_source == "serialized_binding"
                else ("randomized_surface_argument_1", "randomized_surface_argument_2")
            ),
        },
        results=results,
        holm_family_passes=passes,
        protocol_freeze_gate_passes=passes,
        argument_source=argument_source,
    )


def write_raw_leakage_report(report: RawLeakageReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
