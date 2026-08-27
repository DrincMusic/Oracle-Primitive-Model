from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TraceabilityEntry:
    requirement: str
    implementation: str
    tests: tuple[str, ...]
    artifact: str
    status: str


_GROUPED_ENTRIES = (
    TraceabilityEntry(
        "ALG-001..008",
        "rlmgraph.opm.algebra; rlmgraph.opm.generation",
        ("test_opm_algebra.py", "test_opm_generation.py"),
        "generated/dry-run-traces.json",
        "validated_reduced_scale",
    ),
    TraceabilityEntry(
        "REN-001..007",
        "rlmgraph.opm.rendering; rlmgraph.opm.data; rlmgraph.opm.splits",
        ("test_opm_rendering.py", "test_opm_splits.py"),
        "generated/test-recombination.validation.jsonl",
        "validated_reduced_scale",
    ),
    TraceabilityEntry(
        "DAT-001..005",
        "rlmgraph.opm.splits",
        ("test_opm_splits.py",),
        "generated/v1.1.3-conformance.reconstructed.json",
        "v1_1_3_train_validation_conformant_test_splits_blocked",
    ),
    TraceabilityEntry(
        "MOD-001..011",
        "rlmgraph.opm.model; rlmgraph.opm.data",
        ("test_opm_model.py", "test_opm_validation.py"),
        "OPM_V1_STATUS.md",
        "validated_cpu",
    ),
    TraceabilityEntry(
        "ORC-001..006",
        "rlmgraph.opm.model; rlmgraph.opm.probes",
        ("test_opm_model.py", "test_opm_validation.py"),
        "generated/raw-oracle-leakage.v1.1.3.json",
        "v1_1_3_raw_channels_passed_neural_trained_runs_pending",
    ),
    TraceabilityEntry(
        "BAS-001..003",
        "rlmgraph.opm.model; rlmgraph.opm.accounting",
        ("test_opm_model.py", "test_opm_validation.py"),
        "OPM_V1_STATUS.md",
        "parameter_counts_and_cpu_flop_trace_validated",
    ),
    TraceabilityEntry(
        "LOS-001..003; TRN-001..004",
        "rlmgraph.opm.training",
        ("test_opm_validation.py",),
        "OPM_V1_STATUS.md",
        "bounded_validation_only",
    ),
    TraceabilityEntry(
        "MET-001..008; STA-001..007",
        "rlmgraph.opm.evaluation; rlmgraph.opm.statistics; rlmgraph.opm.claims",
        ("test_opm_splits.py", "test_opm_validation.py"),
        "claim-ledger.validation.json",
        "bootstrap_and_claim_ledger_logic_validated_synthetic_only",
    ),
    TraceabilityEntry(
        "SYS-001..004",
        "rlmgraph.opm.cli; rlmgraph.opm.protocol; rlmgraph.opm.sealing",
        ("test_opm_validation.py", "test_opm_reproducibility.py"),
        "OPM_V1_STATUS.md",
        "validated_cpu_primary_gated_locked_evaluator_validated",
    ),
)


def _expand_requirement_expression(expression: str) -> tuple[str, ...]:
    expanded: list[str] = []
    for component in expression.split(";"):
        component = component.strip()
        match = re.fullmatch(r"([A-Z]{3})-(\d{3})(?:\.\.(\d{3}))?", component)
        if match is None:
            raise ValueError(f"invalid requirement expression: {component}")
        prefix, first_text, last_text = match.groups()
        first = int(first_text)
        last = int(last_text) if last_text is not None else first
        if last < first:
            raise ValueError(f"descending requirement expression: {component}")
        expanded.extend(f"{prefix}-{number:03d}" for number in range(first, last + 1))
    return tuple(expanded)


ENTRIES = tuple(
    TraceabilityEntry(
        requirement,
        grouped.implementation,
        grouped.tests,
        grouped.artifact,
        grouped.status,
    )
    for grouped in _GROUPED_ENTRIES
    for requirement in _expand_requirement_expression(grouped.requirement)
)


def write_traceability(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(entry) for entry in ENTRIES], indent=2) + "\n", encoding="utf-8"
    )
