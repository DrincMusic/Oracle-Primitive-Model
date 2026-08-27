from copy import deepcopy
from pathlib import Path

import pytest

from scripts import opm_claim_decision as claims

CLAIM_IDS = (
    "H1-PRIMARY",
    "STA-004-SYMBOLIC-ORACLE",
    "STA-004-RECOMBINATION",
    "STA-004-ACTIVE-ABLATION",
    "STA-004-UNRELATED-PRESERVATION",
    "STA-004-SENTINEL",
    "STA-004-ADAPTER-ONLY",
    "STA-004-RAW-ORACLE-PROBES",
    "STA-004-NEURAL-PROBES",
    "MECHANISM-CAUSALLY-REUSABLE-PRIMITIVE",
    "MET-005-INTERCHANGE-MECHANISM",
    "MET-007-SURFACE-INVARIANCE",
    "H2-THEORETICAL-RESOURCE-ADVANTAGE",
    "H3-REAL-HARDWARE-ADVANTAGE",
    "H4-CORRECT-BOUNDARY-OF-REUSE",
)


def _authorization() -> dict[str, object]:
    return {
        "schema_version": claims.AUTHORIZATION_SCHEMA,
        "authorized_operations": list(claims.AUTHORIZED_OPERATIONS),
        "prohibited_operations": list(claims.PROHIBITED_OPERATIONS),
    }


def _spec() -> dict[str, object]:
    return {
        "rules": [
            {
                "claim_id": claim_id,
                "claim_type": "test",
                "wording": f"locked wording for {claim_id}",
                "rule": f"locked rule for {claim_id}",
            }
            for claim_id in CLAIM_IDS
        ]
    }


def _aggregates() -> tuple[dict[str, object], ...]:
    baseline = {
        "schema_version": "opm-sealed-baseline-metrics-v1",
        "thresholds_applied": False,
        "pooled_split_accuracy": [
            {
                "condition": "OPM_SHARED",
                "split_name": "test-interpolation",
                "accuracy": 0.999,
            },
            {
                "condition": "DOMAIN_GENERALIST",
                "split_name": "test-interpolation",
                "accuracy": 0.997,
            },
            {
                "condition": "PROC_UNTIED",
                "split_name": "test-interpolation",
                "accuracy": 0.998,
            },
            {
                "condition": "OPM_SHARED",
                "split_name": "test-recombination",
                "accuracy": 0.998,
            },
        ],
    }
    effects = {
        "schema_version": "opm-sealed-primary-effects-v1",
        "thresholds_applied": False,
        "paired_two_level_bootstrap": {
            "delta_generalist": 0.4262,
            "delta_generalist_percentile_95_interval": [0.3019, 0.4923],
            "delta_untied": 0.4978,
        },
    }
    details = []
    operations = ("LOOKUP", "REVERSE", "CHAIN", "LIFT")
    domains = ("PROGRAM", "SCENE", "SET")
    for index, operation in enumerate(operations):
        variant = f"operation:{operation}"
        for domain in domains[:2]:
            details.append(
                {
                    "condition": "OPM_SHARED",
                    "family": "ablation",
                    "artifact_split": "test-interpolation",
                    "variant": variant,
                    "target_operation": operation,
                    "target_domain": domain,
                    "accuracy_change": -0.30,
                }
            )
        details.append(
            {
                "condition": "OPM_SHARED",
                "family": "ablation",
                "artifact_split": "test-interpolation",
                "variant": variant,
                "target_operation": operations[(index + 1) % len(operations)],
                "target_domain": domains[2],
                "accuracy_change": 0.0,
            }
        )
    summary = [
        {
            "condition": "OPM_SHARED",
            "family": "adapter-only",
            "artifact_split": split,
            "accuracy": 0.50,
        }
        for split in (
            "test-interpolation",
            "test-recombination",
            "test-renderer",
            "test-structural",
        )
    ]
    summary.extend(
        [
            {
                "condition": "OPM_SHARED",
                "artifact_split": "renderer-pairs",
                "family": "interchange",
                "variant": "cross-domain-renderer-state-swap",
                "accuracy": 0.346,
                "baseline_accuracy": 0.999,
                "accuracy_change": -0.653,
            },
            {
                "condition": "OPM_SHARED",
                "artifact_split": "test-renderer",
                "family": "surface-reversal",
                "variant": "renderer-v2",
                "accuracy": 0.999,
                "baseline_accuracy": 0.999,
                "accuracy_change": 0.0,
            },
        ]
    )
    interventions = {
        "schema_version": "opm-sealed-intervention-metrics-v1",
        "thresholds_applied": False,
        "by_component_domain_operation": details,
        "sentinel_max_absolute_logit_change": [
            {"max_absolute_logit_change": 0.0} for _ in range(80)
        ],
        "summary": summary,
    }
    probe_rows = []
    for seed in (1101, 2202, 3303, 4404, 5505):
        for step in (1, 2):
            failed = seed == 4404 and step == 1
            probe_rows.append(
                {
                    "condition": "OPM_SHARED",
                    "training_seed": seed,
                    "evidence_step": step,
                    "validation_accuracy": 0.51375 if failed else 0.50,
                    "wilson_low": 0.5015 if failed else 0.48,
                    "wilson_high": 0.526 if failed else 0.52,
                    "p_value": 0.014349 if failed else 1.0,
                }
            )
    probes = {
        "schema_version": "opm-sealed-probe-metrics-v1",
        "thresholds_applied": False,
        "results": probe_rows,
    }
    return baseline, effects, interventions, probes


def _evaluate(
    baseline: dict[str, object],
    effects: dict[str, object],
    interventions: dict[str, object],
    probes: dict[str, object],
) -> dict[str, dict[str, object]]:
    decisions = claims.evaluate_claims(
        baseline=baseline,
        effects=effects,
        interventions=interventions,
        probes=probes,
        spec=_spec(),
        gate=claims.CapabilityGate.from_authorization(_authorization()),
    )
    return {str(row["claim_id"]): row for row in decisions}


def test_capability_gate_allows_decisions_and_denies_prohibited_operations() -> None:
    gate = claims.CapabilityGate.from_authorization(_authorization())
    gate.require(claims.Capability.CLAIM_THRESHOLD_APPLICATION)
    for capability in (
        claims.Capability.CHECKPOINT_ACCESS,
        claims.Capability.MODEL_LOADING,
        claims.Capability.PREDICTION_ROW_ACCESS,
        claims.Capability.SEALED_LABEL_ACCESS,
        claims.Capability.PREDICTION_GENERATION,
        claims.Capability.AGGREGATE_RECOMPUTATION,
        claims.Capability.THRESHOLD_MODIFICATION,
        claims.Capability.TRAINING,
        claims.Capability.SCIENTIFIC_NARRATIVE_GENERATION,
    ):
        with pytest.raises(claims.AuthorizationError, match=capability.value):
            gate.require(capability)


def test_source_audit_rejects_model_runtime_import(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed.py"
    allowed.write_text("import json\n", encoding="utf-8")
    claims.assert_claim_source_is_read_only(allowed)
    denied = tmp_path / "denied.py"
    denied.write_text("import torch\n", encoding="utf-8")
    with pytest.raises(claims.AuthorizationError, match="prohibited runtime"):
        claims.assert_claim_source_is_read_only(denied)


def test_claims_are_decided_independently_and_mechanism_failure_does_not_erase_h1() -> None:
    decisions = _evaluate(*_aggregates())
    assert len(decisions) == 15
    assert decisions["H1-PRIMARY"]["status"] == "SUPPORTED"
    assert decisions["STA-004-NEURAL-PROBES"]["status"] == "NOT_SUPPORTED"
    assert decisions["MECHANISM-CAUSALLY-REUSABLE-PRIMITIVE"]["status"] == "NOT_SUPPORTED"
    assert decisions["STA-004-SYMBOLIC-ORACLE"]["status"] == "NOT_EVALUABLE"
    assert decisions["STA-004-RAW-ORACLE-PROBES"]["status"] == "NOT_EVALUABLE"
    assert decisions["MET-005-INTERCHANGE-MECHANISM"]["status"] == "NOT_EVALUABLE"
    assert decisions["MET-007-SURFACE-INVARIANCE"]["status"] == "NOT_EVALUABLE"
    statuses = [row["status"] for row in decisions.values()]
    assert statuses.count("SUPPORTED") == 6
    assert statuses.count("NOT_SUPPORTED") == 2
    assert statuses.count("INCONCLUSIVE") == 0
    assert statuses.count("NOT_EVALUABLE") == 7


@pytest.mark.parametrize(
    ("interval", "opm_accuracy", "expected"),
    [
        ([0.03, 0.08], 0.999, "SUPPORTED"),
        ([0.01, 0.08], 0.999, "INCONCLUSIVE"),
        ([-0.01, 0.02], 0.999, "NOT_SUPPORTED"),
        ([0.03, 0.08], 0.980, "INCONCLUSIVE"),
    ],
)
def test_h1_boundary_rule(interval: list[float], opm_accuracy: float, expected: str) -> None:
    baseline, effects, interventions, probes = deepcopy(_aggregates())
    effects["paired_two_level_bootstrap"]["delta_generalist_percentile_95_interval"] = interval
    baseline["pooled_split_accuracy"][0]["accuracy"] = opm_accuracy
    decisions = _evaluate(baseline, effects, interventions, probes)
    assert decisions["H1-PRIMARY"]["status"] == expected
