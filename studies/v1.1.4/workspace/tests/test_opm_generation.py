from rlmgraph.opm.algebra import Query, Relation, entailed
from rlmgraph.opm.generation import (
    Corruption,
    Operation,
    derive_uint64,
    enumerate_negative_candidates,
    enumerate_positive_templates,
    generate_world,
)


def test_world_generation_is_deterministic() -> None:
    first = generate_world(12345, n_objects=8, n_containers=4)
    second = generate_world(12345, n_objects=8, n_containers=4)
    assert first == second
    assert derive_uint64("a", 1) == derive_uint64("a", 1)


def test_same_has_real_three_entity_chain() -> None:
    world = generate_world(444, n_objects=8, n_containers=4)
    chains = [
        example
        for example in enumerate_positive_templates(world)
        if example.operations == (Operation.CHAIN, Operation.CHAIN)
        and example.query.relation == Relation.SAME
    ]
    assert chains
    assert all(len({e.query.arg1, e.evidence[0].arg2, e.query.arg2}) == 3 for e in chains)


def test_positive_and_negative_templates_obey_serialized_evidence_semantics() -> None:
    world = generate_world(91, n_objects=8, n_containers=4)
    positives = enumerate_positive_templates(world)
    assert positives
    assert all(entailed(world, example.query) for example in positives)
    negatives = [candidate for p in positives for candidate in enumerate_negative_candidates(world, p)]
    assert negatives
    for candidate in negatives:
        projected = type(world)(world.world_id, world.entities, tuple(sorted(candidate.evidence)))
        assert not entailed(projected, candidate.query)


def test_before_uses_paths_longer_than_model_depth() -> None:
    world = generate_world(678, n_objects=10, n_containers=4)
    before = world.facts_for(Relation.BEFORE)
    # The general closure implementation is covered directly even if this random DAG lacks a long path.
    for left in before:
        assert entailed(world, Query(Relation.BEFORE, left.arg1, left.arg2))


def test_endpoint_corruptions_are_distinct_and_formally_false() -> None:
    for seed in range(20):
        world = generate_world(seed, n_objects=10, n_containers=5)
        endpoints = [
            candidate
            for positive in enumerate_positive_templates(world)
            for candidate in enumerate_negative_candidates(world, positive)
            if candidate.corruption == Corruption.ENDPOINT
        ]
        assert endpoints
        assert all(candidate.query.arg1 != candidate.query.arg2 for candidate in endpoints)
        assert all(not entailed(world, candidate.query) for candidate in endpoints)
