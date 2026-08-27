from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from rlmgraph.opm.artifacts import load_checkpoint, run_id, save_checkpoint, write_run_manifest
from rlmgraph.opm.data import Vocabulary, collate, materialize
from rlmgraph.opm.generation import enumerate_positive_templates, generate_world
from rlmgraph.opm.model import ModelConfig, ModelKind, OPMModel
from rlmgraph.opm.protocol import Lifecycle, ProtocolState
from rlmgraph.opm.readiness import implementation_readiness
from rlmgraph.opm.rendering import Domain
from rlmgraph.opm.sealing import (
    audit_label_access,
    evaluate_sealed_predictions,
    seal_validation_labels,
)


def _batch():
    world = generate_world(1101, n_objects=10, n_containers=5)
    examples = enumerate_positive_templates(world)[:4]
    records = [materialize(world, example, Domain.SET, 0) for example in examples]
    return collate(records, Vocabulary.build()), records


def _step(model: OPMModel, optimizer: torch.optim.Optimizer, batch) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(batch), batch.labels)
    loss.backward()
    optimizer.step()


def test_run_identity_and_manifest_are_reproducible(tmp_path: Path) -> None:
    arguments = {
        "configuration": {"model": "OPM_SHARED"},
        "specification_version": "1.0.0",
        "dataset_fingerprints": {"train": "abc"},
        "model_seed": 1101,
        "code_revision": "deadbeef",
    }
    assert run_id(**arguments) == run_id(**arguments)
    assert run_id(**arguments) != run_id(**{**arguments, "model_seed": 2202})
    identifier, path = write_run_manifest(
        tmp_path,
        configuration=arguments["configuration"],
        dataset_fingerprints=arguments["dataset_fingerprints"],
        model_seed=1101,
        code_revision="deadbeef",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == identifier
    assert payload["primary_run"] is False


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    batch, _ = _batch()
    torch.manual_seed(1101)
    model = OPMModel(ModelConfig(len(Vocabulary.build().tokens), dropout=0.0), ModelKind.OPM_SHARED, 1101)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    _step(model, optimizer, batch)
    checkpoint = tmp_path / "checkpoint.pt"
    digest = save_checkpoint(checkpoint, model=model, optimizer=optimizer, step=1, metadata={"validation_only": True})
    assert digest == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _step(model, optimizer, batch)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    resumed, resumed_optimizer, step, metadata = load_checkpoint(
        checkpoint, optimizer_factory=lambda parameters: torch.optim.AdamW(parameters, lr=3e-4)
    )
    _step(resumed, resumed_optimizer, batch)
    assert step == 1 and metadata["validation_only"] is True
    assert all(torch.equal(expected[name], value) for name, value in resumed.state_dict().items())


def test_validation_label_seal_separates_labels_and_audits_access(tmp_path: Path) -> None:
    _, records = _batch()
    manifest = seal_validation_labels(records, tmp_path, "small")
    assert '"label"' not in Path(manifest.input_path).read_text(encoding="utf-8")
    assert '"label"' in Path(manifest.label_path).read_text(encoding="utf-8")
    audit = tmp_path / "access.jsonl"
    audit_label_access(audit, actor="test", purpose="validation", label_path=Path(manifest.label_path))
    assert json.loads(audit.read_text(encoding="utf-8"))["actor"] == "test"


def test_locked_evaluator_denies_label_access_before_protocol_freeze(tmp_path: Path) -> None:
    _, records = _batch()
    seal_validation_labels(records, tmp_path, "small")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("", encoding="utf-8")
    audit = tmp_path / "locked-evaluator.audit.jsonl"
    with pytest.raises(PermissionError, match="protocol is frozen"):
        evaluate_sealed_predictions(
            tmp_path / "small.seal-manifest.json",
            predictions,
            audit,
            protocol=ProtocolState(),
            actor="test",
            purpose="denial validation",
        )
    entry = json.loads(audit.read_text(encoding="utf-8"))
    assert entry["decision"] == "DENIED"


@pytest.mark.parametrize(
    ("label_access", "aggregate_evaluation"),
    ((False, False), (True, False), (False, True)),
)
def test_locked_evaluator_requires_both_authorizations_after_freeze(
    tmp_path: Path, label_access: bool, aggregate_evaluation: bool
) -> None:
    _, records = _batch()
    seal_validation_labels(records, tmp_path, "small")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("", encoding="utf-8")
    with pytest.raises(PermissionError, match="requires separate sealed-label-access"):
        evaluate_sealed_predictions(
            tmp_path / "small.seal-manifest.json",
            predictions,
            tmp_path / "locked-evaluator.audit.jsonl",
            protocol=ProtocolState(
                lifecycle=Lifecycle.PROTOCOL_FROZEN,
                protocol_frozen=True,
                sealed_label_access_authorized=label_access,
                aggregate_test_evaluation_authorized=aggregate_evaluation,
            ),
            actor="test",
            purpose="post-freeze denial validation",
        )


def test_stage1_is_fail_closed_pending_training_erratum() -> None:
    state = ProtocolState(
        lifecycle=Lifecycle.PRIMARY_RUNS,
        protocol_frozen=True,
        primary_runs_authorized=True,
        trained_probes_authorized=True,
        prediction_generation_authorized=True,
        sealed_label_access_authorized=False,
        aggregate_test_evaluation_authorized=False,
        claim_decisions_authorized=False,
        canonical_training_authorized=False,
        execution_status="BLOCKED_PENDING_V1_1_4_APPROVAL",
    )
    with pytest.raises(PermissionError, match="primary experiments are prohibited"):
        state.require_primary()
    with pytest.raises(PermissionError, match="primary experiments are prohibited"):
        state.require_trained_probes()
    with pytest.raises(PermissionError, match="primary experiments are prohibited"):
        state.require_prediction_generation()
    with pytest.raises(PermissionError, match="locked evaluation requires"):
        state.require_locked_evaluation()


def test_locked_evaluator_validation_harness_enforces_exact_keys(tmp_path: Path) -> None:
    _, records = _batch()
    seal_validation_labels(records, tmp_path, "small")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps({"example_id": record.example_id, "prediction": record.label}) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    result = evaluate_sealed_predictions(
        tmp_path / "small.seal-manifest.json",
        predictions,
        tmp_path / "locked-evaluator.audit.jsonl",
        protocol=ProtocolState(),
        actor="test",
        purpose="validation harness",
        validation_harness=True,
    )
    assert result.accuracy == 1.0 and result.count == len(records)
    assert not hasattr(result, "labels")

    duplicate = json.loads(predictions.read_text(encoding="utf-8").splitlines()[0])
    predictions.write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate prediction key"):
        evaluate_sealed_predictions(
            tmp_path / "small.seal-manifest.json",
            predictions,
            tmp_path / "locked-evaluator.audit.jsonl",
            protocol=ProtocolState(),
            actor="test",
            purpose="duplicate validation",
            validation_harness=True,
        )


def test_readiness_is_fail_closed_for_primary_runs(tmp_path: Path) -> None:
    report = implementation_readiness(tmp_path)
    assert report.ready_for_protocol_freeze is False
    assert report.primary_runs_authorized is False
    assert any(not check.passed for check in report.checks)
    trained = next(check for check in report.checks if check.name == "trained_neural_probes")
    assert trained.passed is False
    assert trained.blocking_for_protocol_freeze is False
