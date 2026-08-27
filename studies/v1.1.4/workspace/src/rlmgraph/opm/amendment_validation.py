from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EndpointDiagnostic:
    maximum_total_variation: float
    mean_total_variation: float
    compared_strata: int
    largest_strata: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class AmendmentConformanceReport:
    amendment_version: str
    passed: bool
    train_rows: int
    validation_rows: int
    relation_histograms_matched: bool
    query_argument_histograms_matched: bool
    binding_collisions: int
    evidence_position_histograms_matched: bool
    joint_evidence_position_histograms_matched: bool
    endpoint_diagnostics: dict[str, EndpointDiagnostic]
    endpoint_diagnostics_are_gates: bool
    test_labels_accessed: bool


def _load(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _type(binding_id: int) -> str:
    return "OBJECT" if binding_id < 32 else "CONTAINER"


def _argument_histograms_matched(
    tagged_rows: list[tuple[str, dict[str, object]]],
) -> bool:
    histograms: dict[tuple[object, ...], dict[int, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for split, row in tagged_rows:
        label = int(row["label"])
        operation = int(row["operation_ids"][0])
        relation = str(row["query"][0])
        for position, value in enumerate(row["argument_entity_ids"]):
            key = (split, row["domain"], operation, relation, position, _type(int(value)))
            histograms[key][label][int(value)] += 1
    return all(values[0] == values[1] for values in histograms.values())


def _evidence_histograms_matched(
    tagged_rows: list[tuple[str, dict[str, object]]], *, joint: bool
) -> bool:
    histograms: dict[tuple[object, ...], dict[int, Counter[object]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for split, row in tagged_rows:
        key = (
            split,
            row["domain"],
            int(row["operation_ids"][0]),
            str(row["query"][0]),
        )
        label = int(row["label"])
        positions = tuple(int(value) for value in row["evidence_indices"])
        if joint:
            histograms[key][label][positions] += 1
        else:
            for step, position in enumerate(positions):
                histograms[key + (step,)][label][position] += 1
    return all(values[0] == values[1] for values in histograms.values())


def _endpoint_diagnostic(rows: list[dict[str, object]]) -> EndpointDiagnostic:
    histograms: dict[tuple[object, ...], dict[int, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for row in rows:
        operation = int(row["operation_ids"][0])
        for fact_position, evidence_index in enumerate(row["evidence_indices"]):
            if int(evidence_index) < 0:
                continue
            fact = row["facts"][int(evidence_index)]
            endpoints = row["fact_endpoint_ids"][int(evidence_index)]
            for endpoint_position, binding_id in enumerate(endpoints):
                key = (
                    row["domain"],
                    operation,
                    fact_position,
                    endpoint_position,
                    _type(int(binding_id)),
                    fact[0],
                )
                histograms[key][int(row["label"])][int(binding_id)] += 1
    scored: list[tuple[float, tuple[object, ...], int, int]] = []
    for key, labels in histograms.items():
        positive_total = sum(labels[1].values())
        negative_total = sum(labels[0].values())
        if not positive_total or not negative_total:
            continue
        support = set(labels[0]) | set(labels[1])
        variation = 0.5 * sum(
            abs(labels[1][value] / positive_total - labels[0][value] / negative_total)
            for value in support
        )
        scored.append((variation, key, positive_total, negative_total))
    scored.sort(reverse=True)
    return EndpointDiagnostic(
        maximum_total_variation=scored[0][0] if scored else 0.0,
        mean_total_variation=(sum(item[0] for item in scored) / len(scored) if scored else 0.0),
        compared_strata=len(scored),
        largest_strata=tuple(
            {
                "total_variation": variation,
                "domain": key[0],
                "operation": key[1],
                "selected_fact_position": key[2],
                "endpoint_position": key[3],
                "entity_type": key[4],
                "fact_relation": key[5],
                "positive_count": positive_total,
                "negative_count": negative_total,
            }
            for variation, key, positive_total, negative_total in scored[:10]
        ),
    )


def validate_amended_data(train_path: Path, validation_path: Path) -> AmendmentConformanceReport:
    train = _load(train_path)
    validation = _load(validation_path)
    tagged = [("train", row) for row in train] + [
        ("validation", row) for row in validation
    ]
    relation_counts: dict[tuple[object, ...], dict[int, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for split, row in tagged:
        key = (split, row["domain"], int(row["operation_ids"][0]))
        relation_counts[key][int(row["label"])][str(row["query"][0])] += 1
    relation_matched = all(values[0] == values[1] for values in relation_counts.values())
    argument_matched = _argument_histograms_matched(tagged)
    evidence_matched = _evidence_histograms_matched(tagged, joint=False)
    joint_evidence_matched = _evidence_histograms_matched(tagged, joint=True)
    collisions = sum(
        int(row["argument_entity_ids"][0] == row["argument_entity_ids"][1])
        for _, row in tagged
    )
    return AmendmentConformanceReport(
        amendment_version=("1.1.3" if "1.1.3" in train_path.name else "1.1.2"),
        passed=(
            relation_matched
            and argument_matched
            and evidence_matched
            and joint_evidence_matched
            and collisions == 0
        ),
        train_rows=len(train),
        validation_rows=len(validation),
        relation_histograms_matched=relation_matched,
        query_argument_histograms_matched=argument_matched,
        binding_collisions=collisions,
        evidence_position_histograms_matched=evidence_matched,
        joint_evidence_position_histograms_matched=joint_evidence_matched,
        endpoint_diagnostics={
            "train": _endpoint_diagnostic(train),
            "validation": _endpoint_diagnostic(validation),
        },
        endpoint_diagnostics_are_gates=False,
        test_labels_accessed=False,
    )


def write_amendment_report(report: AmendmentConformanceReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
