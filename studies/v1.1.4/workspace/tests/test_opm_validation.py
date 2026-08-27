from dataclasses import replace

import numpy as np
import pytest

from rlmgraph.opm.accounting import account
from rlmgraph.opm.data import Vocabulary, collate, materialize
from rlmgraph.opm.evaluation import ablation_logits, adapter_only_logits
from rlmgraph.opm.generation import enumerate_positive_templates, generate_world
from rlmgraph.opm.model import ModelConfig, ModelKind, OPMModel
from rlmgraph.opm.probes import (
    ProbeResult,
    RawLeakageProbeResult,
    categorical_leakage_probe,
    holm_fail_to_reject,
    logistic_leakage_probe,
    raw_probe_family_passes,
    wilson_interval,
)
from rlmgraph.opm.protocol import CURRENT_PROTOCOL, ProtocolState
from rlmgraph.opm.rendering import Domain
from rlmgraph.opm.training import TrainingConfig, train_validation


def _single_batch():
    world = generate_world(771, n_objects=10, n_containers=5)
    example = next(item for item in enumerate_positive_templates(world) if len(item.operations) == 2)
    vocabulary = Vocabulary.build()
    return collate([materialize(world, example, Domain.SET, 0)], vocabulary), vocabulary


def test_primary_run_is_hard_gated() -> None:
    with pytest.raises(PermissionError, match="primary experiments are prohibited"):
        ProtocolState().require_primary()
    CURRENT_PROTOCOL.require_primary()
    with pytest.raises(PermissionError, match="locked evaluation requires separate"):
        CURRENT_PROTOCOL.require_locked_evaluation()


def test_validation_training_is_bounded() -> None:
    batch, vocabulary = _single_batch()
    model = OPMModel(ModelConfig(len(vocabulary.tokens), dropout=0.0), ModelKind.OPM_SHARED, 1101)
    result = train_validation(model, [batch], TrainingConfig(), validation_step_limit=1)
    assert result.steps == 1
    assert np.isfinite(result.final_loss)
    with pytest.raises(ValueError, match="limited"):
        train_validation(model, [batch], TrainingConfig(), validation_step_limit=101)


def test_accounting_and_interventions() -> None:
    batch, vocabulary = _single_batch()
    model = OPMModel(ModelConfig(len(vocabulary.tokens), dropout=0.0), ModelKind.OPM_SHARED, 1101)
    model.eval()
    report = account(model)
    assert report.total_parameters > report.active_primitive_parameters_one_step
    active = model.operation_to_primitive[int(batch.operation_ids[0, 0])]
    baseline = model(batch)
    ablated = ablation_logits(model, batch, active)
    adapter_only = adapter_only_logits(model, batch)
    assert baseline.shape == ablated.shape == adapter_only.shape == (1, 2)


def test_probe_and_holm_contract() -> None:
    rng = np.random.default_rng(9)
    train_x = rng.normal(size=(200, 3))
    train_y = rng.integers(0, 2, size=200)
    valid_x = rng.normal(size=(200, 3))
    valid_y = rng.integers(0, 2, size=200)
    result = logistic_leakage_probe(train_x, train_y, valid_x, valid_y)
    assert 0 <= result.accuracy <= 1
    safe = ProbeResult(0.5, 0.5, False, True)
    assert holm_fail_to_reject([safe, safe, safe])
    leaking = ProbeResult(0.8, 1e-8, True, False)
    assert not holm_fail_to_reject([safe, leaking])


def test_categorical_probe_and_wilson_contract() -> None:
    train_x = [[str(index % 2)] for index in range(100)]
    train_y = np.asarray([index % 2 for index in range(100)])
    validation_x = [[str(index % 2)] for index in range(40)]
    validation_y = np.asarray([index % 2 for index in range(40)])
    result = categorical_leakage_probe(
        train_x, train_y, validation_x, validation_y, channel="deliberate_leak"
    )
    assert result.accuracy == 1.0
    assert result.rejects_null is True
    low, high = wilson_interval(20, 40)
    assert low < 0.5 < high


def test_raw_probe_family_is_fail_closed() -> None:
    passing = RawLeakageProbeResult(
        "a", 0.5, 50, 100, 0.4, 0.6, True, 0.5, False, True, True
    )
    family = [passing, replace(passing, channel="b"), replace(passing, channel="c")]
    assert raw_probe_family_passes(family)
    family[0] = replace(family[0], p_value=0.001, rejects_null=True)
    assert not raw_probe_family_passes(family)
