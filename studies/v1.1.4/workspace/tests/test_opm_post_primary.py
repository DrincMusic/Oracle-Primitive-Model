import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from rlmgraph.opm.data import Vocabulary
from rlmgraph.opm.model import ModelConfig, ModelKind, OPMBatch, OPMModel
from scripts import opm_post_primary as stage1
from scripts.opm_post_primary_executor import _group_pairs_by_fact_shape

FAKE_COMMIT = "c" * 40


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stage1.canonical_json_bytes(payload))


@dataclass
class Fixture:
    workspace: Path
    transition: Path
    checkpoints: Path
    evaluation: Path
    probe: Path
    intervention: Path
    resources: Path
    output: Path
    transition_sha256: str


def _make_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    monkeypatch.setattr(stage1, "git_commit", lambda _workspace: FAKE_COMMIT)
    workspace = tmp_path
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "executor.py"
    source.write_text("# frozen executor fixture\n", encoding="utf-8")
    matrix = workspace / "primary-matrix.json"
    matrix.write_text('{"state":"COMPLETED"}\n', encoding="utf-8")
    entries = []
    for condition in stage1.DECLARED_CONDITIONS:
        for seed in stage1.DECLARED_SEEDS:
            run_id = f"{condition.lower()}-{seed}"
            checkpoint = workspace / "primary" / run_id / "checkpoints" / "step-00500.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"checkpoint:{condition}:{seed}".encode())
            checkpoint_sha256 = stage1.sha256_file(checkpoint)
            reconciliation = {"checkpoint_sha256": checkpoint_sha256}
            entries.append(
                {
                    "condition": condition,
                    "training_seed": seed,
                    "run_id": run_id,
                    "selected_step": 500,
                    "checkpoint_path": checkpoint.relative_to(workspace).as_posix(),
                    "checkpoint_artifact_id": f"{run_id}:step-00500",
                    "checkpoint_sha256": checkpoint_sha256,
                    "reconciliation": reconciliation,
                    "reconciliation_record_sha256": stage1.canonical_sha256(reconciliation),
                }
            )
    checkpoint_manifest = {
        "schema_version": stage1.CHECKPOINT_MANIFEST_SCHEMA,
        "primary_matrix_sha256": stage1.sha256_file(matrix),
        "checkpoint_count": 20,
        "entries": entries,
    }
    checkpoints = workspace / "checkpoints.json"
    _write_json(checkpoints, checkpoint_manifest)

    split_specs = []
    for split in stage1.TEST_SPLITS:
        data = workspace / "data" / f"{split}.jsonl"
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text(json.dumps({"example_id": f"{split}-1"}) + "\n", encoding="utf-8")
        manifest = workspace / "data" / f"{split}.manifest.json"
        _write_json(
            manifest,
            {
                "split": split,
                "row_count": 1,
                "sha256": stage1.sha256_file(data),
                "labels_separated": True,
            },
        )
        split_specs.append(
            {
                "split_name": split,
                "input_path": data.relative_to(workspace).as_posix(),
                "input_sha256": stage1.sha256_file(data),
                "input_manifest_path": manifest.relative_to(workspace).as_posix(),
                "input_manifest_sha256": stage1.sha256_file(manifest),
                "row_count": 1,
            }
        )
    evaluation = workspace / "evaluation.json"
    _write_json(
        evaluation,
        {"schema_version": stage1.EVALUATION_SPEC_SCHEMA, "splits": split_specs},
    )

    probe_inputs = {}
    for split in ("train", "validation"):
        data = workspace / "data" / f"{split}.jsonl"
        data.write_text(
            json.dumps({"example_id": f"{split}-1", "label": 0}) + "\n",
            encoding="utf-8",
        )
        manifest = workspace / "data" / f"{split}.manifest.json"
        _write_json(
            manifest,
            {
                "row_count": 1,
                "sha256": stage1.sha256_file(data),
                "labels_separated": False,
            },
        )
        probe_inputs[split] = {
            "path": data.relative_to(workspace).as_posix(),
            "sha256": stage1.sha256_file(data),
            "manifest_path": manifest.relative_to(workspace).as_posix(),
            "manifest_sha256": stage1.sha256_file(manifest),
            "row_count": 1,
        }
    probe = workspace / "probe.json"
    _write_json(
        probe,
        {"schema_version": stage1.PROBE_SPEC_SCHEMA, "inputs": probe_inputs},
    )
    intervention = workspace / "intervention.json"
    _write_json(intervention, {"schema_version": stage1.INTERVENTION_SPEC_SCHEMA})

    resources = workspace / "resources.json"
    _write_json(
        resources,
        {
            "schema_version": stage1.RESOURCE_MANIFEST_SCHEMA,
            "sealed_labels_mounted": False,
            "prohibited_resources": sorted(stage1.SEALED_PATH_MARKERS),
            "readable_roots": ["."],
            "writable_root": "outputs",
        },
    )
    source_records = [
        {"path": source.relative_to(workspace).as_posix(), "sha256": stage1.sha256_file(source)}
    ]
    transition_payload = {
        "schema_version": stage1.TRANSITION_SCHEMA,
        "from_state": "PRIMARY_TRAINING_COMPLETE",
        "to_state": "POST_TRAINING_STAGE_1_AUTHORIZED",
        "protocol_version": stage1.PROTOCOL_VERSION,
        "protocol_commit": FAKE_COMMIT,
        "executor_source_commit": FAKE_COMMIT,
        "executor_sources": source_records,
        "executor_source_aggregate_sha256": stage1.canonical_sha256(source_records),
        "primary_matrix_sha256": stage1.sha256_file(matrix),
        "selected_checkpoint_manifest_sha256": stage1.sha256_file(checkpoints),
        "selected_checkpoint_count": 20,
        "evaluation_spec_sha256": stage1.sha256_file(evaluation),
        "probe_spec_sha256": stage1.sha256_file(probe),
        "intervention_spec_sha256": stage1.sha256_file(intervention),
        "authorized_operations": list(stage1.AUTHORIZED_OPERATIONS),
        "prohibited_operations": list(stage1.PROHIBITED_OPERATIONS),
        "sealed_label_access_authorized": False,
        "aggregate_test_evaluation_authorized": False,
        "claim_decisions_authorized": False,
    }
    transition = workspace / "transition.json"
    _write_json(transition, transition_payload)
    return Fixture(
        workspace=workspace,
        transition=transition,
        checkpoints=checkpoints,
        evaluation=evaluation,
        probe=probe,
        intervention=intervention,
        resources=resources,
        output=workspace / "outputs" / "run",
        transition_sha256=stage1.sha256_file(transition),
    )


def _resign(fixture: Fixture, **hash_fields: Path) -> None:
    transition = stage1.read_json(fixture.transition, "transition")
    for field, path in hash_fields.items():
        transition[field] = stage1.sha256_file(path)
    _write_json(fixture.transition, transition)
    fixture.transition_sha256 = stage1.sha256_file(fixture.transition)


def _preflight(fixture: Fixture):
    return stage1.preflight_stage1(
        workspace=fixture.workspace,
        transition_path=fixture.transition,
        expected_transition_sha256=fixture.transition_sha256,
        checkpoint_manifest_path=fixture.checkpoints,
        evaluation_spec_path=fixture.evaluation,
        probe_spec_path=fixture.probe,
        intervention_spec_path=fixture.intervention,
        resource_manifest_path=fixture.resources,
        output_root=fixture.output,
    )


def test_preflight_accepts_exact_twenty_checkpoint_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    preflight, gate, checkpoints, inventories = _preflight(fixture)
    assert preflight.state == "PASS"
    assert preflight.checkpoint_count == 20
    assert len(checkpoints) == 20
    assert len(inventories) == 4
    gate.require(stage1.Capability.LABEL_BLIND_PREDICTION)


def test_missing_or_wrong_transition_hash_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    fixture.transition.unlink()
    with pytest.raises(stage1.AuthorizationError, match="missing or invalid"):
        _preflight(fixture)

    fixture = _make_fixture(tmp_path / "wrong", monkeypatch)
    with pytest.raises(stage1.AuthorizationError, match="missing or invalid"):
        stage1.preflight_stage1(
            workspace=fixture.workspace,
            transition_path=fixture.transition,
            expected_transition_sha256="0" * 64,
            checkpoint_manifest_path=fixture.checkpoints,
            evaluation_spec_path=fixture.evaluation,
            probe_spec_path=fixture.probe,
            intervention_spec_path=fixture.intervention,
            resource_manifest_path=fixture.resources,
            output_root=fixture.output,
        )


@pytest.mark.parametrize("fault", ["nineteen", "duplicate", "wrong-step", "unapproved"])
def test_checkpoint_matrix_faults_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    manifest = stage1.read_json(fixture.checkpoints, "checkpoints")
    if fault == "nineteen":
        manifest["entries"].pop()
        manifest["checkpoint_count"] = 19
    elif fault == "duplicate":
        manifest["entries"][-1] = dict(manifest["entries"][0])
    elif fault == "wrong-step":
        manifest["entries"][0]["selected_step"] = 1000
    else:
        manifest["entries"][0]["condition"] = "UNAPPROVED"
    _write_json(fixture.checkpoints, manifest)
    _resign(fixture, selected_checkpoint_manifest_sha256=fixture.checkpoints)
    with pytest.raises(stage1.IntegrityError):
        _preflight(fixture)


def test_altered_checkpoint_bytes_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    manifest = stage1.read_json(fixture.checkpoints, "checkpoints")
    checkpoint = fixture.workspace / manifest["entries"][0]["checkpoint_path"]
    checkpoint.write_bytes(b"altered")
    with pytest.raises(stage1.IntegrityError, match="altered or missing"):
        _preflight(fixture)


def test_label_field_and_sealed_resource_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    evaluation = stage1.read_json(fixture.evaluation, "evaluation")
    split = evaluation["splits"][0]
    data = fixture.workspace / split["input_path"]
    data.write_text(json.dumps({"example_id": "x", "label": 1}) + "\n", encoding="utf-8")
    manifest_path = fixture.workspace / split["input_manifest_path"]
    manifest = stage1.read_json(manifest_path, "manifest")
    manifest["sha256"] = stage1.sha256_file(data)
    _write_json(manifest_path, manifest)
    split["input_sha256"] = stage1.sha256_file(data)
    split["input_manifest_sha256"] = stage1.sha256_file(manifest_path)
    _write_json(fixture.evaluation, evaluation)
    _resign(fixture, evaluation_spec_sha256=fixture.evaluation)
    with pytest.raises(stage1.AuthorizationError, match="target/scoring field"):
        _preflight(fixture)

    fixture = _make_fixture(tmp_path / "sealed", monkeypatch)
    sealed = fixture.workspace / "sealed-labels"
    sealed.mkdir()
    supplied = sealed / "transition.json"
    supplied.write_bytes(fixture.transition.read_bytes())
    with pytest.raises(stage1.AuthorizationError, match="sealed-label resource"):
        stage1.preflight_stage1(
            workspace=fixture.workspace,
            transition_path=supplied,
            expected_transition_sha256=stage1.sha256_file(supplied),
            checkpoint_manifest_path=fixture.checkpoints,
            evaluation_spec_path=fixture.evaluation,
            probe_spec_path=fixture.probe,
            intervention_spec_path=fixture.intervention,
            resource_manifest_path=fixture.resources,
            output_root=fixture.output,
        )


def test_resume_requires_exact_complete_job_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    _preflight(fixture)
    resumed, _gate, _checkpoints, _inventories = _preflight(fixture)
    assert resumed.resume is True
    intervention = stage1.read_json(fixture.intervention, "intervention")
    intervention["changed"] = True
    _write_json(fixture.intervention, intervention)
    _resign(fixture, intervention_spec_sha256=fixture.intervention)
    with pytest.raises(stage1.IntegrityError, match="job specification"):
        _preflight(fixture)


def test_denied_aggregate_and_claim_capabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    transition = stage1.read_json(fixture.transition, "transition")
    gate = stage1.CapabilityGate.from_transition(transition, fixture.transition_sha256)
    with pytest.raises(stage1.AuthorizationError, match="aggregate_test_scoring"):
        stage1.refuse_aggregate_evaluation(gate)
    with pytest.raises(stage1.AuthorizationError, match="claim_threshold_application"):
        stage1.refuse_claim_decision(gate)


def test_probe_optimizer_and_base_model_change_abort() -> None:
    torch.manual_seed(7)
    model = OPMModel(
        ModelConfig(
            vocabulary_size=len(Vocabulary.build().tokens),
            d_model=16,
            d_entity=8,
            d_domain=4,
            d_hidden_primitive=32,
            encoder_ff=32,
            heads=4,
            encoder_layers=1,
        ),
        ModelKind.OPM_SHARED,
        1101,
    )
    stage1.freeze_base_model(model)
    probe = torch.nn.Linear(16, 2)
    bad_optimizer = torch.optim.AdamW([*probe.parameters(), *model.parameters()], lr=1e-3)
    with pytest.raises(stage1.AuthorizationError, match="not restricted"):
        stage1.assert_probe_optimizer_scope(bad_optimizer, probe, model)
    before = stage1.tensor_state_sha256(model)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(stage1.AuthorizationError, match="base model changed"):
        stage1.assert_base_model_unchanged(model, before)


def _prediction(example_id: str, logits: list[float]) -> dict[str, object]:
    return {
        "artifact_schema_version": stage1.PREDICTION_SCHEMA,
        "condition": "OPM_SHARED",
        "training_seed": 1101,
        "example_id": example_id,
        "prediction": 0,
        "logits_or_scores": logits,
    }


def test_prediction_writer_rejects_missing_duplicate_and_nan(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jsonl"
    writer = stage1.AtomicJsonlArtifact(
        missing_path,
        artifact_kind="prediction",
        expected_rows=2,
        expected_example_ids={"a", "b"},
    )
    writer.write(_prediction("a", [1.0, 0.0]))
    with pytest.raises(stage1.ReconciliationError, match="row count"):
        writer.commit({})
    assert not missing_path.exists()

    with stage1.AtomicJsonlArtifact(
        tmp_path / "duplicate.jsonl", artifact_kind="prediction", expected_rows=2
    ) as writer:
        writer.write(_prediction("a", [1.0, 0.0]))
        with pytest.raises(stage1.ReconciliationError, match="duplicate"):
            writer.write(_prediction("a", [1.0, 0.0]))

    with (
        stage1.AtomicJsonlArtifact(
            tmp_path / "nan.jsonl", artifact_kind="prediction", expected_rows=1
        ) as writer,
        pytest.raises(stage1.ReconciliationError, match="NaN or infinite"),
    ):
        writer.write(_prediction("a", [float("nan"), 0.0]))


def _gate() -> stage1.CapabilityGate:
    transition = {
        "schema_version": stage1.TRANSITION_SCHEMA,
        "authorized_operations": list(stage1.AUTHORIZED_OPERATIONS),
        "prohibited_operations": list(stage1.PROHIBITED_OPERATIONS),
    }
    return stage1.CapabilityGate.from_transition(transition, "a" * 64)


def test_partial_freeze_refused_and_frozen_bundle_immutable(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    for name in ("environment.json", "job-spec.json", "preflight.json"):
        _write_json(output / name, {"name": name})
    (output / "execution.log.jsonl").write_text("{}\n", encoding="utf-8")
    artifact = output / "predictions" / "one.jsonl"
    with stage1.AtomicJsonlArtifact(
        artifact,
        artifact_kind="prediction",
        expected_rows=1,
        expected_example_ids={"a"},
    ) as writer:
        writer.write(_prediction("a", [1.0, 0.0]))
        manifest = writer.commit(
            {
                "condition": "OPM_SHARED",
                "training_seed": 1101,
                "checkpoint_sha256": "1" * 64,
                "model_pre_sha256": "2" * 64,
                "model_post_sha256": "2" * 64,
            }
        )
    checkpoints = [
        {"condition": condition, "training_seed": seed, "checkpoint_sha256": "1" * 64}
        for condition in stage1.DECLARED_CONDITIONS
        for seed in stage1.DECLARED_SEEDS
    ]
    expected = {
        "path": artifact.relative_to(output).as_posix(),
        "artifact_kind": "prediction",
        "row_count": 1,
        "identity_set_sha256": manifest["identity_set_sha256"],
        "condition": "OPM_SHARED",
        "training_seed": 1101,
        "checkpoint_sha256": "1" * 64,
    }
    with pytest.raises(stage1.IntegrityError, match="missing artifact sidecar"):
        stage1.reconcile_and_freeze(
            output_root=output,
            transition_sha256="a" * 64,
            checkpoints=checkpoints,
            expected_artifacts=[expected, {"path": "missing.jsonl"}],
            environment_path=output / "environment.json",
            execution_log_path=output / "execution.log.jsonl",
            gate=_gate(),
        )
    reconciliation, freeze = stage1.reconcile_and_freeze(
        output_root=output,
        transition_sha256="a" * 64,
        checkpoints=checkpoints,
        expected_artifacts=[expected],
        environment_path=output / "environment.json",
        execution_log_path=output / "execution.log.jsonl",
        gate=_gate(),
    )
    assert reconciliation["state"] == "PASS"
    assert freeze["state"] == "ARTIFACTS_RECONCILED_AND_FROZEN"
    with pytest.raises(stage1.AuthorizationError, match="cannot be modified"):
        stage1.reconcile_and_freeze(
            output_root=output,
            transition_sha256="a" * 64,
            checkpoints=checkpoints,
            expected_artifacts=[expected],
            environment_path=output / "environment.json",
            execution_log_path=output / "execution.log.jsonl",
            gate=_gate(),
        )


def test_small_forward_fixture_is_bit_deterministic() -> None:
    torch.manual_seed(11)
    config = ModelConfig(
        vocabulary_size=len(Vocabulary.build().tokens),
        d_model=16,
        d_entity=8,
        d_domain=4,
        d_hidden_primitive=32,
        encoder_ff=32,
        heads=4,
        encoder_layers=1,
    )
    model = OPMModel(config, ModelKind.OPM_SHARED, 1101)
    stage1.freeze_base_model(model)
    batch = OPMBatch(
        fact_tokens=torch.ones((2, 8, 12), dtype=torch.long),
        fact_token_mask=torch.ones((2, 8, 12), dtype=torch.bool),
        query_tokens=torch.ones((2, 12), dtype=torch.long),
        query_token_mask=torch.ones((2, 12), dtype=torch.bool),
        domain_ids=torch.tensor([0, 1]),
        argument_entity_ids=torch.tensor([[0, 1], [2, 3]]),
        fact_endpoint_ids=torch.zeros((2, 8, 2), dtype=torch.long),
        evidence_indices=torch.tensor([[0, 1], [2, 3]]),
        operation_ids=torch.tensor([[0, 1], [2, 3]]),
        step_mask=torch.ones((2, 2), dtype=torch.bool),
        labels=None,
    )
    with torch.inference_mode():
        first = model(batch)
        second = model(batch)
    assert torch.equal(first, second)


def test_interchange_batches_are_partitioned_by_source_and_target_fact_shape() -> None:
    pairs = [
        ({"rendered_facts": ["a"] * 8}, {"rendered_facts": ["b"] * 8}),
        ({"rendered_facts": ["a"] * 12}, {"rendered_facts": ["b"] * 8}),
        ({"rendered_facts": ["a"] * 8}, {"rendered_facts": ["b"] * 12}),
    ]
    grouped = _group_pairs_by_fact_shape(pairs)
    assert sorted(grouped) == [(8, 8), (8, 12), (12, 8)]
    assert all(len(group) == 1 for group in grouped.values())
