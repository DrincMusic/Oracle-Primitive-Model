import pytest
import torch

from rlmgraph.opm.data import Vocabulary, collate, materialize
from rlmgraph.opm.generation import (
    enumerate_negative_candidates,
    enumerate_positive_templates,
    generate_world,
)
from rlmgraph.opm.model import ModelConfig, ModelKind, OPMModel, parameter_counts
from rlmgraph.opm.rendering import Domain


def _batch():
    world = generate_world(8080, n_objects=10, n_containers=5)
    positives = enumerate_positive_templates(world)
    positive = next(example for example in positives if len(example.operations) == 2)
    negative = next(
        candidate
        for source in positives
        for candidate in enumerate_negative_candidates(world, source)
        if len(candidate.operations) == 2
    )
    records = [
        materialize(world, positive, Domain.SET, 0),
        materialize(world, negative, Domain.SCENE, 1),
    ]
    vocabulary = Vocabulary.build()
    return collate(records, vocabulary), vocabulary


@pytest.mark.parametrize("kind", list(ModelKind))
def test_all_approved_models_forward_and_backward(kind: ModelKind) -> None:
    torch.manual_seed(1101)
    batch, vocabulary = _batch()
    model = OPMModel(ModelConfig(len(vocabulary.tokens), dropout=0.0), kind, model_seed=1101)
    logits, states = model(batch, return_states=True)
    assert logits.shape == (2, 2)
    assert len(states) == 3
    loss = torch.nn.functional.cross_entropy(logits, batch.labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert parameter_counts(model)["trainable"] > 0


def test_fixed_permutation_is_deterministic_and_uses_four_modules() -> None:
    vocabulary = Vocabulary.build()
    config = ModelConfig(len(vocabulary.tokens))
    first = OPMModel(config, ModelKind.OPM_SHARED, model_seed=2202)
    second = OPMModel(config, ModelKind.OPM_SHARED, model_seed=2202)
    assert first.operation_to_primitive == second.operation_to_primitive
    assert len(set(first.operation_to_primitive)) == 4


def test_clone_control_starts_with_equal_domain_copies() -> None:
    vocabulary = Vocabulary.build()
    model = OPMModel(ModelConfig(len(vocabulary.tokens)), ModelKind.PROC_CLONE, 3303)
    width = model.config.primitive_count
    for index in range(width):
        left = model.primitives[index].state_dict()
        right = model.primitives[width + index].state_dict()
        assert all(torch.equal(left[name], right[name]) for name in left)
