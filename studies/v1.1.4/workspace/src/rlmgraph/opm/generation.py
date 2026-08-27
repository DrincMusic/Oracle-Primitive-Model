from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

from .algebra import Entity, EntityType, Fact, Query, Relation, World, entailed, validate_world


class Operation(IntEnum):
    LOOKUP = 0
    REVERSE = 1
    CHAIN = 2
    LIFT = 3


class Procedure(IntEnum):
    DIRECT = 0
    SYMMETRY = 1
    TRANSITIVE = 2
    LIFT = 3


class Corruption(IntEnum):
    NONE = 0
    DIRECT_MISMATCH = 1
    ENDPOINT = 2
    MIDDLE = 3
    ORDER = 4
    RELATION = 5


@dataclass(frozen=True)
class Example:
    world_id: int
    query: Query
    label: int
    evidence: tuple[Fact, ...]
    operations: tuple[Operation, ...]
    procedure: Procedure
    corruption: Corruption = Corruption.NONE

    def canonical_key(self) -> tuple[object, ...]:
        return (self.world_id, self.query, self.operations, self.evidence, self.corruption)


def derive_uint64(*parts: object) -> int:
    joined = "/".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(joined).digest()[:8], "big", signed=False)


def _rng(*parts: object) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(derive_uint64(*parts)))


def _sample_pairs(
    rng: np.random.Generator, candidates: Sequence[tuple[int, int]], count: int
) -> list[tuple[int, int]]:
    indices = rng.permutation(len(candidates))[:count]
    return [candidates[int(index)] for index in indices]


def generate_world(seed: int, n_objects: int | None = None, n_containers: int | None = None) -> World:
    size_rng = _rng(seed, "size")
    n_objects = n_objects or int(size_rng.integers(6, 11))
    n_containers = n_containers or int(size_rng.integers(3, 6))
    if n_objects < 3 or n_containers < 2:
        raise ValueError("world requires at least three objects and two containers")
    objects = list(range(n_objects))
    containers = list(range(n_objects, n_objects + n_containers))
    entities = tuple(
        [Entity(i, EntityType.OBJECT) for i in objects]
        + [Entity(i, EntityType.CONTAINER) for i in containers]
    )
    for attempt in range(100):
        rng = _rng(seed, "attempt", attempt)
        topo = [int(value) for value in rng.permutation(objects)]
        before_candidates = sorted(
            (topo[i], topo[j]) for i in range(len(topo)) for j in range(i + 1, len(topo))
        )
        before = _sample_pairs(rng, before_candidates, min(n_objects, 8))
        same_members = sorted(int(value) for value in rng.choice(objects, size=3, replace=False))
        same = [(same_members[0], same_members[1]), (same_members[1], same_members[2])]
        link_candidates = sorted(
            (left, right) for left in objects for right in objects if left < right
        )
        links = _sample_pairs(rng, link_candidates, min(n_objects, 8))
        direct = [(obj, int(rng.choice(containers))) for obj in sorted(objects)]
        root = int(rng.choice(containers))
        nested = [(container, root) for container in sorted(containers) if container != root]
        facts = tuple(
            sorted(
                [Fact(Relation.DIRECT_IN, *pair) for pair in direct]
                + [Fact(Relation.NESTED_IN, *pair) for pair in nested]
                + [Fact(Relation.BEFORE, *pair) for pair in before]
                + [Fact(Relation.SAME, *pair) for pair in same]
                + [Fact(Relation.LINK, *pair) for pair in links]
            )
        )
        world = World(world_id=seed, entities=entities, facts=facts)
        try:
            validate_world(world)
        except ValueError:
            continue
        if enumerate_positive_templates(world):
            return world
    raise RuntimeError(f"failed to generate valid world after 100 attempts: {seed}")


def enumerate_positive_templates(world: World) -> tuple[Example, ...]:
    examples: list[Example] = []
    for fact in world.facts:
        if fact.relation == Relation.DIRECT_IN:
            query = Query(Relation.WITHIN, fact.arg1, fact.arg2)
        elif fact.relation in (Relation.BEFORE, Relation.SAME, Relation.LINK):
            query = Query(fact.relation, fact.arg1, fact.arg2)
        else:
            continue
        examples.append(
            Example(world.world_id, query, 1, (fact,), (Operation.LOOKUP,), Procedure.DIRECT)
        )
    for fact in world.facts:
        if fact.relation in (Relation.SAME, Relation.LINK) and fact.arg1 != fact.arg2:
            examples.append(
                Example(
                    world.world_id,
                    Query(fact.relation, fact.arg2, fact.arg1),
                    1,
                    (fact,),
                    (Operation.REVERSE,),
                    Procedure.SYMMETRY,
                )
            )
    for relation in (Relation.BEFORE, Relation.SAME):
        facts = world.facts_for(relation)
        by_left: dict[int, list[Fact]] = {}
        for fact in facts:
            by_left.setdefault(fact.arg1, []).append(fact)
        for first in facts:
            for second in sorted(by_left.get(first.arg2, ())):
                if len({first.arg1, first.arg2, second.arg2}) != 3:
                    continue
                examples.append(
                    Example(
                        world.world_id,
                        Query(relation, first.arg1, second.arg2),
                        1,
                        (first, second),
                        (Operation.CHAIN, Operation.CHAIN),
                        Procedure.TRANSITIVE,
                    )
                )
    nested_by_child = {fact.arg1: fact for fact in world.facts_for(Relation.NESTED_IN)}
    for direct in world.facts_for(Relation.DIRECT_IN):
        nested = nested_by_child.get(direct.arg2)
        if nested is not None:
            examples.append(
                Example(
                    world.world_id,
                    Query(Relation.WITHIN, direct.arg1, nested.arg2),
                    1,
                    (direct, nested),
                    (Operation.LIFT, Operation.LIFT),
                    Procedure.LIFT,
                )
            )
    unique = {example.canonical_key(): example for example in examples if entailed(world, example.query)}
    return tuple(sorted(unique.values(), key=lambda item: item.canonical_key()))


def enumerate_negative_candidates(world: World, positive: Example) -> tuple[Example, ...]:
    candidates: list[Example] = []
    types = world.entity_types
    required_type = types[positive.query.arg2]
    for entity_id, entity_type in sorted(types.items()):
        if (
            entity_type == required_type
            and entity_id != positive.query.arg2
            and entity_id != positive.query.arg1
        ):
            candidates.append(
                replace(
                    positive,
                    query=Query(positive.query.relation, positive.query.arg1, entity_id),
                    label=0,
                    corruption=Corruption.ENDPOINT,
                )
            )
    if positive.query.relation in (Relation.BEFORE, Relation.WITHIN):
        swapped = Query(positive.query.relation, positive.query.arg2, positive.query.arg1)
        if types.get(swapped.arg1) == types.get(positive.query.arg1):
            candidates.append(
                replace(positive, query=swapped, label=0, corruption=Corruption.ORDER)
            )
    if len(positive.evidence) == 2:
        second = positive.evidence[1]
        for fact in world.facts_for(second.relation):
            if fact.arg1 != positive.evidence[0].arg2:
                candidates.append(
                    replace(
                        positive,
                        evidence=(positive.evidence[0], fact),
                        label=0,
                        corruption=Corruption.MIDDLE,
                    )
                )
    if len(positive.evidence) == 1:
        selected = positive.evidence[0]
        for fact in world.facts_for(selected.relation):
            if fact != selected:
                candidates.append(
                    replace(
                        positive,
                        evidence=(fact,),
                        label=0,
                        corruption=Corruption.DIRECT_MISMATCH,
                    )
                )
    valid = {
        candidate.canonical_key(): candidate
        for candidate in candidates
        if not entailed(
            World(world.world_id, world.entities, tuple(sorted(candidate.evidence))),
            candidate.query,
        )
        and not (
            candidate.corruption == Corruption.ENDPOINT
            and entailed(world, candidate.query)
        )
    }
    for candidate in valid.values():
        if candidate.corruption == Corruption.ENDPOINT:
            if candidate.query.arg1 == candidate.query.arg2:
                raise AssertionError("self-endpoint corruption escaped filtering")
            if entailed(world, candidate.query):
                raise AssertionError("endpoint corruption is true in the formal world")
    return tuple(sorted(valid.values(), key=lambda item: item.canonical_key()))


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
