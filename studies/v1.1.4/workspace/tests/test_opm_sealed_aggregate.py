import json
from pathlib import Path

import numpy as np
import pytest

from scripts import opm_sealed_aggregate as aggregate


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(aggregate.canonical_json_bytes(value))


def _authorization() -> dict[str, object]:
    return {
        "schema_version": aggregate.AUTHORIZATION_SCHEMA,
        "authorized_operations": list(aggregate.AUTHORIZED_OPERATIONS),
        "prohibited_operations": list(aggregate.PROHIBITED_OPERATIONS),
        "sealed_label_access_authorized": True,
        "aggregate_test_evaluation_authorized": True,
        "claim_decisions_authorized": False,
    }


def test_capability_gate_allows_aggregate_and_denies_claims() -> None:
    gate = aggregate.CapabilityGate.from_authorization(_authorization())
    gate.require(aggregate.Capability.SEALED_TARGET_READ)
    gate.require(aggregate.Capability.BOOTSTRAP)
    with pytest.raises(aggregate.AuthorizationError, match="claim_threshold_application"):
        gate.require(aggregate.Capability.CLAIM_THRESHOLD_APPLICATION)
    with pytest.raises(aggregate.AuthorizationError, match="claim_decision"):
        gate.require(aggregate.Capability.CLAIM_DECISION)


def test_source_audit_rejects_model_runtime_import(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed.py"
    allowed.write_text("import json\n", encoding="utf-8")
    aggregate.assert_evaluator_source_is_aggregate_only(allowed)
    denied = tmp_path / "denied.py"
    denied.write_text("from rlmgraph.opm.model import OPMModel\n", encoding="utf-8")
    with pytest.raises(aggregate.AuthorizationError, match="prohibited model runtime"):
        aggregate.assert_evaluator_source_is_aggregate_only(denied)


def test_locked_spec_construction_does_not_require_sealed_target_files(tmp_path: Path) -> None:
    freeze_path = tmp_path / "stage1" / "freeze.json"
    freeze = {
        "schema_version": aggregate.STAGE1_FREEZE_SCHEMA,
        "state": aggregate.STAGE1_STATE,
        "aggregate_test_evaluation_performed": False,
        "merkle_root": "1" * 64,
    }
    _write_json(freeze_path, freeze)
    split_specs = []
    for split in aggregate.SPLITS:
        data = tmp_path / "data" / f"{split}.jsonl"
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text(json.dumps({"example_id": split}) + "\n", encoding="utf-8")
        manifest = tmp_path / "data" / f"{split}.manifest.json"
        _write_json(
            manifest,
            {
                "split": split,
                "labels_separated": True,
                "labels_sha256": "2" * 64,
                "row_count": 1,
            },
        )
        split_specs.append(
            {
                "split_name": split,
                "input_path": data.relative_to(tmp_path).as_posix(),
                "input_sha256": aggregate.sha256_file(data),
                "input_manifest_path": manifest.relative_to(tmp_path).as_posix(),
                "input_manifest_sha256": aggregate.sha256_file(manifest),
                "row_count": 1,
            }
        )
    evaluation = tmp_path / "evaluation.json"
    _write_json(evaluation, {"splits": split_specs})
    spec = aggregate.build_locked_aggregate_spec(
        workspace=tmp_path,
        stage1_freeze_path=freeze_path,
        expected_stage1_freeze_sha256=aggregate.sha256_file(freeze_path),
        stage1_evaluation_spec_path=evaluation,
    )
    assert len(spec["sealed_targets"]) == 4
    assert all(not (tmp_path / target["path"]).exists() for target in spec["sealed_targets"])


def test_sealed_target_read_is_denied_before_file_open(tmp_path: Path) -> None:
    target = tmp_path / "sealed-labels" / "test.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text('{"example_id":"e","label":1}\n', encoding="utf-8")
    gate = aggregate.CapabilityGate()
    resources = aggregate.ResourcePolicy(
        workspace=tmp_path,
        frozen_root=tmp_path / "frozen",
        metadata_paths=frozenset(),
        target_paths=frozenset({target.resolve()}),
        output_root=tmp_path / "output",
        gate=gate,
    )
    spec = {
        "sealed_targets": [
            {
                "split_name": "test-recombination",
                "path": target.relative_to(tmp_path).as_posix(),
                "sha256": aggregate.sha256_file(target),
                "row_count": 1,
            }
        ]
    }
    metadata = {
        "test-recombination": {
            "e": aggregate.ExampleMetadata("test-recombination", 1, "SET", 0, "REVERSE", 1)
        }
    }
    audit = tmp_path / "audit.jsonl"
    with pytest.raises(aggregate.AuthorizationError, match="sealed_target_read"):
        aggregate.load_sealed_targets(
            workspace=tmp_path,
            spec=spec,
            metadata=metadata,
            resources=resources,
            audit_path=audit,
        )
    assert not audit.exists()


def test_two_level_bootstrap_is_deterministic_and_paired() -> None:
    def with_counts(values: list[list[float]]) -> np.ndarray:
        correct = np.asarray(values)
        return np.stack((correct, np.ones_like(correct)), axis=-1)

    world_scores: dict[int, dict[str, np.ndarray]] = {}
    for model_seed in aggregate.SEEDS:
        world_scores[model_seed] = {
            "OPM_SHARED": with_counts([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
            "DOMAIN_GENERALIST": with_counts([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "PROC_UNTIED": with_counts([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        }
    first = aggregate.paired_two_level_bootstrap(world_scores, replicates=10_000, seed=99117)
    second = aggregate.paired_two_level_bootstrap(world_scores, replicates=10_000, seed=99117)
    assert first == second
    assert first["replicates"] == 10_000
    assert first["delta_generalist"] == pytest.approx(0.5)
    assert first["delta_untied"] == pytest.approx(2 / 3)


def test_world_macro_weights_rows_within_each_cell() -> None:
    values = np.asarray(
        [
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[0.0, 9.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    assert aggregate._world_macro_accuracy(values) == pytest.approx(0.7)


def test_recombination_bootstrap_condition_inventory_matches_evaluator() -> None:
    inventory = {
        model_seed: {
            condition: {} for condition in ("OPM_SHARED", "DOMAIN_GENERALIST", "PROC_UNTIED")
        }
        for model_seed in aggregate.SEEDS
    }
    for model_seed in aggregate.SEEDS:
        for condition in aggregate.CONDITIONS:
            should_collect = condition in inventory[model_seed]
            assert should_collect is (condition != "PROC_CLONE")


def test_compute_aggregates_populates_recombination_worlds(tmp_path: Path) -> None:
    recombination = {}
    for index in range(100):
        recombination[f"p{index}"] = aggregate.ExampleMetadata(
            "test-recombination", index, "PROGRAM", 0, "CHAIN", 2
        )
        recombination[f"s{index}"] = aggregate.ExampleMetadata(
            "test-recombination", 100 + index, "SCENE", 0, "LIFT", 2
        )
        recombination[f"t{index}"] = aggregate.ExampleMetadata(
            "test-recombination", 200 + index, "SET", 0, "REVERSE", 2
        )
    metadata = {
        "test-recombination": recombination,
        "test-interpolation": {
            "i": aggregate.ExampleMetadata("test-interpolation", 8, "SET", 1, "LOOKUP", 1)
        },
        "test-renderer": {
            "r": aggregate.ExampleMetadata("test-renderer", 9, "SCENE", 2, "CHAIN", 2)
        },
        "test-structural": {
            "u": aggregate.ExampleMetadata("test-structural", 10, "PROGRAM", 0, "LIFT", 2)
        },
    }
    global_metadata = {
        example_id: (split, example)
        for split, examples in metadata.items()
        for example_id, example in examples.items()
    }
    labels = {
        split: {example_id: 1 for example_id in examples} for split, examples in metadata.items()
    }
    artifacts = []
    for condition in aggregate.CONDITIONS:
        for model_seed in aggregate.SEEDS:
            for split, examples in metadata.items():
                path = (
                    tmp_path / "predictions" / condition / f"seed-{model_seed}" / f"{split}.jsonl"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                for example_id in examples:
                    prediction = 1 if condition == "OPM_SHARED" else 0
                    rows.append(
                        {
                            "condition": condition,
                            "training_seed": model_seed,
                            "example_id": example_id,
                            "prediction": prediction,
                            "logits_or_scores": [-1.0, 1.0] if prediction else [1.0, -1.0],
                        }
                    )
                path.write_bytes(b"".join(aggregate.canonical_json_bytes(row) for row in rows))
                artifacts.append(
                    {
                        "path": path.relative_to(tmp_path).as_posix(),
                        "artifact_kind": "prediction",
                        "row_count": len(rows),
                    }
                )
    freeze = {"artifacts": artifacts}
    gate = aggregate.CapabilityGate.from_authorization(_authorization())
    baseline, effects, interventions, probes = aggregate.compute_aggregates(
        stage_root=tmp_path,
        freeze=freeze,
        metadata=metadata,
        global_metadata=global_metadata,
        labels=labels,
        gate=gate,
    )
    assert baseline["prediction_rows_joined"] == 6_060
    assert effects["paired_two_level_bootstrap"]["delta_generalist"] == pytest.approx(1.0)
    assert len(effects["per_seed_effects"]) == 5
    assert interventions["intervention_rows_joined"] == 0
    assert probes["results"] == []
