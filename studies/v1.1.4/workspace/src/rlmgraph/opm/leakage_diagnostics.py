from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .probes import categorical_leakage_probe, categorical_predictions
from .raw_leakage import _ENTITY_ALIAS
from .splits import DiagnosticSplitConfig, SplitName, allocate_split


@dataclass(frozen=True)
class LeakageDiagnostics:
    canonical_gate_unchanged: bool
    test_labels_accessed: bool
    validation_accuracy: float
    accuracy_by_domain: dict[str, float]
    accuracy_by_operation: dict[str, float]
    accuracy_by_argument_position: dict[str, float]
    equal_world_weight_accuracy: float
    world_cluster_bootstrap_interval: tuple[float, float]
    validation_worlds: int
    alias_label_rate_correlation_train_to_validation: dict[str, float]
    interpretation: str


@dataclass(frozen=True)
class ReplicaResult:
    replica: int
    seed_namespace: str
    rows: int
    worlds: int
    canonical_world_overlap: int
    accuracy: float
    wilson_interval: tuple[float, float]
    p_value: float
    crosses_canonical_first_holm_threshold: bool
    alias_rate_correlations: dict[str, float]


@dataclass(frozen=True)
class LeakageReplicaReport:
    diagnostic_only: bool
    canonical_gate_unchanged: bool
    test_labels_accessed: bool
    replica_count: int
    results: tuple[ReplicaResult, ...]
    mean_accuracy: float
    replicas_crossing_canonical_first_holm_threshold: int
    interpretation: str


def _load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _aliases(rows: list[dict[str, object]]) -> list[list[str]]:
    result = [_ENTITY_ALIAS.findall(str(row["rendered_query"])) for row in rows]
    if any(len(item) != 2 for item in result):
        raise ValueError("every query must contain two randomized aliases")
    return result


def _stratified_accuracy(
    predictions: np.ndarray, labels: np.ndarray, strata: list[str]
) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for prediction, label, stratum in zip(predictions, labels, strata, strict=True):
        grouped[stratum].append(bool(prediction == label))
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def _alias_rate_correlation(
    train_aliases: list[list[str]],
    train_labels: np.ndarray,
    validation_aliases: list[list[str]],
    validation_labels: np.ndarray,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for position in (0, 1):
        rates: list[dict[str, float]] = []
        for aliases, labels in (
            (train_aliases, train_labels),
            (validation_aliases, validation_labels),
        ):
            sums: dict[str, int] = defaultdict(int)
            counts: dict[str, int] = defaultdict(int)
            for alias, label in zip(aliases, labels, strict=True):
                sums[alias[position]] += int(label)
                counts[alias[position]] += 1
            rates.append({key: sums[key] / counts[key] for key in counts})
        shared = sorted(set(rates[0]) & set(rates[1]))
        output[f"argument_{position + 1}"] = float(
            np.corrcoef(
                [rates[0][key] for key in shared], [rates[1][key] for key in shared]
            )[0, 1]
        )
    return output


def diagnose_argument_leakage(
    train_path: Path, validation_path: Path, *, bootstrap_replicates: int = 10_000
) -> LeakageDiagnostics:
    train_rows = _load_rows(train_path)
    validation_rows = _load_rows(validation_path)
    train_aliases = _aliases(train_rows)
    validation_aliases = _aliases(validation_rows)
    train_labels = np.asarray([row["label"] for row in train_rows], dtype=np.int64)
    validation_labels = np.asarray([row["label"] for row in validation_rows], dtype=np.int64)
    predictions = categorical_predictions(train_aliases, train_labels, validation_aliases)
    position_results = {
        f"argument_{position + 1}": categorical_leakage_probe(
            [[item[position]] for item in train_aliases],
            train_labels,
            [[item[position]] for item in validation_aliases],
            validation_labels,
            channel=f"argument_{position + 1}",
        ).accuracy
        for position in (0, 1)
    }
    correctness = predictions == validation_labels
    by_world: dict[int, list[bool]] = defaultdict(list)
    for row, correct in zip(validation_rows, correctness, strict=True):
        by_world[int(row["world_id"])].append(bool(correct))
    world_ids = sorted(by_world)
    equal_world_accuracy = float(np.mean([np.mean(by_world[key]) for key in world_ids]))
    rng = np.random.Generator(np.random.PCG64DXSM(91001))
    bootstrap = np.empty(bootstrap_replicates)
    world_arrays = [np.asarray(by_world[key], dtype=np.float64) for key in world_ids]
    for replicate in range(bootstrap_replicates):
        sampled = rng.integers(0, len(world_arrays), size=len(world_arrays))
        values = np.concatenate([world_arrays[int(index)] for index in sampled])
        bootstrap[replicate] = values.mean()
    interval = tuple(float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))
    correlation = _alias_rate_correlation(
        train_aliases, train_labels, validation_aliases, validation_labels
    )
    aligned = max(abs(value) for value in correlation.values()) >= 0.2
    interpretation = (
        "train-to-validation alias-rate alignment detected; fixed-sample versus generator cause unresolved"
        if aligned
        else "no strong train-to-validation alias-rate alignment; clustered fluctuation remains plausible"
    )
    return LeakageDiagnostics(
        canonical_gate_unchanged=True,
        test_labels_accessed=False,
        validation_accuracy=float(correctness.mean()),
        accuracy_by_domain=_stratified_accuracy(
            predictions, validation_labels, [str(row["domain"]) for row in validation_rows]
        ),
        accuracy_by_operation=_stratified_accuracy(
            predictions,
            validation_labels,
            [str(row["operation_ids"][0]) for row in validation_rows],
        ),
        accuracy_by_argument_position=position_results,
        equal_world_weight_accuracy=equal_world_accuracy,
        world_cluster_bootstrap_interval=interval,
        validation_worlds=len(world_ids),
        alias_label_rate_correlation_train_to_validation=correlation,
        interpretation=interpretation,
    )


def write_leakage_diagnostics(report: LeakageDiagnostics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def diagnose_replica_namespaces(
    train_path: Path, canonical_validation_path: Path, *, replica_count: int = 5
) -> LeakageReplicaReport:
    if replica_count < 1 or replica_count > 10:
        raise ValueError("diagnostic replica count must be in 1..10")
    train_rows = _load_rows(train_path)
    train_aliases = _aliases(train_rows)
    train_labels = np.asarray([row["label"] for row in train_rows], dtype=np.int64)
    canonical_worlds = {
        int(row["world_id"]) for row in _load_rows(canonical_validation_path)
    }
    results: list[ReplicaResult] = []
    for replica in range(replica_count):
        namespace = f"opm-v1-diagnostic/argument-leakage/replica-{replica + 1}"
        records = allocate_split(DiagnosticSplitConfig(SplitName.VALIDATION, namespace))
        rows = [
            {
                "rendered_query": record.rendered_query,
                "label": record.label,
                "world_id": record.world_id,
            }
            for record in records
        ]
        aliases = _aliases(rows)
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        probe = categorical_leakage_probe(
            train_aliases,
            train_labels,
            aliases,
            labels,
            channel=f"diagnostic_replica_{replica + 1}",
        )
        worlds = {int(row["world_id"]) for row in rows}
        results.append(
            ReplicaResult(
                replica=replica + 1,
                seed_namespace=namespace,
                rows=len(rows),
                worlds=len(worlds),
                canonical_world_overlap=len(worlds & canonical_worlds),
                accuracy=probe.accuracy,
                wilson_interval=(probe.wilson_low, probe.wilson_high),
                p_value=probe.p_value,
                crosses_canonical_first_holm_threshold=probe.p_value < 0.05 / 3,
                alias_rate_correlations=_alias_rate_correlation(
                    train_aliases, train_labels, aliases, labels
                ),
            )
        )
    crossing = sum(item.crosses_canonical_first_holm_threshold for item in results)
    accuracies = [item.accuracy for item in results]
    if crossing >= 2:
        interpretation = "gate-level leakage recurs; deterministic design coupling is plausible"
    elif all(accuracy > 0.5 for accuracy in accuracies) and np.mean(accuracies) >= 0.503:
        interpretation = (
            "gate-level failures are inconsistent, but a weak upward bias recurs across every namespace"
        )
    else:
        interpretation = "canonical failure does not consistently recur; fixed-realization imbalance is plausible"
    return LeakageReplicaReport(
        diagnostic_only=True,
        canonical_gate_unchanged=True,
        test_labels_accessed=False,
        replica_count=replica_count,
        results=tuple(results),
        mean_accuracy=float(np.mean(accuracies)),
        replicas_crossing_canonical_first_holm_threshold=crossing,
        interpretation=interpretation,
    )


def write_replica_report(report: LeakageReplicaReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
