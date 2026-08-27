from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .probes import categorical_leakage_probe, categorical_predictions


@dataclass(frozen=True)
class EvidenceLeakageDiagnostics:
    canonical_gate_unchanged: bool
    test_labels_accessed: bool
    validation_accuracy: float
    accuracy_by_domain: dict[str, float]
    accuracy_by_operation: dict[str, float]
    accuracy_by_query_relation: dict[str, float]
    accuracy_by_renderer_variant: dict[str, float]
    accuracy_by_step_count: dict[str, float]
    accuracy_by_evidence_step_alone: dict[str, float]
    equal_world_weight_accuracy: float
    world_cluster_bootstrap_interval: tuple[float, float]
    validation_worlds: int
    position_label_rate_correlation_train_to_validation: dict[str, float]
    maximum_validation_position_label_rate_gap: float
    largest_validation_position_gaps: tuple[dict[str, object], ...]
    interpretation: str


def _load(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _features(rows: list[dict[str, object]]) -> list[list[str]]:
    return [
        [
            str(row["evidence_indices"][0]),
            str(row["evidence_indices"][1]) if row["step_mask"][1] else "ABSENT",
        ]
        for row in rows
    ]


def _stratified(
    predictions: np.ndarray, labels: np.ndarray, strata: list[str]
) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for prediction, label, stratum in zip(predictions, labels, strata, strict=True):
        grouped[stratum].append(bool(prediction == label))
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def _position_rates(
    rows: list[dict[str, object]], labels: np.ndarray, step: int
) -> dict[str, float]:
    totals: Counter[str] = Counter()
    positives: Counter[str] = Counter()
    for row, label in zip(rows, labels, strict=True):
        value = str(row["evidence_indices"][step]) if row["step_mask"][step] else "ABSENT"
        totals[value] += 1
        positives[value] += int(label)
    return {key: positives[key] / totals[key] for key in totals}


def diagnose_evidence_leakage(
    train_path: Path, validation_path: Path, *, bootstrap_replicates: int = 10_000
) -> EvidenceLeakageDiagnostics:
    train_rows = _load(train_path)
    validation_rows = _load(validation_path)
    train_features = _features(train_rows)
    validation_features = _features(validation_rows)
    train_labels = np.asarray([row["label"] for row in train_rows], dtype=np.int64)
    validation_labels = np.asarray([row["label"] for row in validation_rows], dtype=np.int64)
    predictions = categorical_predictions(train_features, train_labels, validation_features)
    correctness = predictions == validation_labels
    step_accuracy = {
        f"evidence_step_{step + 1}": categorical_leakage_probe(
            [[feature[step]] for feature in train_features],
            train_labels,
            [[feature[step]] for feature in validation_features],
            validation_labels,
            channel=f"evidence_step_{step + 1}",
        ).accuracy
        for step in (0, 1)
    }
    by_world: dict[int, list[bool]] = defaultdict(list)
    for row, correct in zip(validation_rows, correctness, strict=True):
        by_world[int(row["world_id"])].append(bool(correct))
    world_ids = sorted(by_world)
    world_arrays = [np.asarray(by_world[key], dtype=np.float64) for key in world_ids]
    equal_world = float(np.mean([values.mean() for values in world_arrays]))
    rng = np.random.Generator(np.random.PCG64DXSM(91002))
    bootstrap = np.empty(bootstrap_replicates)
    for replicate in range(bootstrap_replicates):
        selected = rng.integers(0, len(world_arrays), size=len(world_arrays))
        bootstrap[replicate] = np.concatenate(
            [world_arrays[int(index)] for index in selected]
        ).mean()
    interval = tuple(float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))
    correlations: dict[str, float] = {}
    gaps: list[tuple[float, int, str, float, float]] = []
    for step in (0, 1):
        train_rates = _position_rates(train_rows, train_labels, step)
        validation_rates = _position_rates(validation_rows, validation_labels, step)
        shared = sorted(set(train_rates) & set(validation_rates))
        correlations[f"evidence_step_{step + 1}"] = float(
            np.corrcoef(
                [train_rates[key] for key in shared],
                [validation_rates[key] for key in shared],
            )[0, 1]
        )
        gaps.extend(
            (
                abs(validation_rates[key] - 0.5),
                step + 1,
                key,
                train_rates[key],
                validation_rates[key],
            )
            for key in shared
        )
    gaps.sort(reverse=True)
    recurring = max(abs(value) for value in correlations.values()) >= 0.2
    interpretation = (
        "train-to-validation position-rate alignment detected; deterministic ordering coupling is plausible"
        if recurring
        else "no strong train-to-validation position-rate alignment; fixed-realization imbalance is plausible"
    )
    return EvidenceLeakageDiagnostics(
        canonical_gate_unchanged=True,
        test_labels_accessed=False,
        validation_accuracy=float(correctness.mean()),
        accuracy_by_domain=_stratified(
            predictions, validation_labels, [str(row["domain"]) for row in validation_rows]
        ),
        accuracy_by_operation=_stratified(
            predictions,
            validation_labels,
            [str(row["operation_ids"][0]) for row in validation_rows],
        ),
        accuracy_by_query_relation=_stratified(
            predictions, validation_labels, [str(row["query"][0]) for row in validation_rows]
        ),
        accuracy_by_renderer_variant=_stratified(
            predictions,
            validation_labels,
            [str(row["renderer_variant"]) for row in validation_rows],
        ),
        accuracy_by_step_count=_stratified(
            predictions,
            validation_labels,
            ["2" if row["step_mask"][1] else "1" for row in validation_rows],
        ),
        accuracy_by_evidence_step_alone=step_accuracy,
        equal_world_weight_accuracy=equal_world,
        world_cluster_bootstrap_interval=interval,
        validation_worlds=len(world_ids),
        position_label_rate_correlation_train_to_validation=correlations,
        maximum_validation_position_label_rate_gap=gaps[0][0],
        largest_validation_position_gaps=tuple(
            {
                "absolute_gap_from_half": gap,
                "evidence_step": step,
                "position": position,
                "train_positive_rate": train_rate,
                "validation_positive_rate": validation_rate,
            }
            for gap, step, position, train_rate, validation_rate in gaps[:10]
        ),
        interpretation=interpretation,
    )


def write_evidence_diagnostics(report: EvidenceLeakageDiagnostics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
