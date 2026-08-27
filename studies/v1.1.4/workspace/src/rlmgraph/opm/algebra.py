from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum


class EntityType(IntEnum):
    OBJECT = 0
    CONTAINER = 1


class Relation(IntEnum):
    DIRECT_IN = 0
    NESTED_IN = 1
    BEFORE = 2
    SAME = 3
    LINK = 4
    WITHIN = 5


FACT_SIGNATURES: dict[Relation, tuple[EntityType, EntityType]] = {
    Relation.DIRECT_IN: (EntityType.OBJECT, EntityType.CONTAINER),
    Relation.NESTED_IN: (EntityType.CONTAINER, EntityType.CONTAINER),
    Relation.BEFORE: (EntityType.OBJECT, EntityType.OBJECT),
    Relation.SAME: (EntityType.OBJECT, EntityType.OBJECT),
    Relation.LINK: (EntityType.OBJECT, EntityType.OBJECT),
}
QUERY_SIGNATURES: dict[Relation, tuple[EntityType, EntityType]] = {
    Relation.WITHIN: (EntityType.OBJECT, EntityType.CONTAINER),
    Relation.BEFORE: (EntityType.OBJECT, EntityType.OBJECT),
    Relation.SAME: (EntityType.OBJECT, EntityType.OBJECT),
    Relation.LINK: (EntityType.OBJECT, EntityType.OBJECT),
}


@dataclass(frozen=True, order=True)
class Entity:
    id: int
    type: EntityType


@dataclass(frozen=True, order=True)
class Fact:
    relation: Relation
    arg1: int
    arg2: int


@dataclass(frozen=True, order=True)
class Query:
    relation: Relation
    arg1: int
    arg2: int


@dataclass(frozen=True)
class World:
    world_id: int
    entities: tuple[Entity, ...]
    facts: tuple[Fact, ...]

    @property
    def entity_types(self) -> dict[int, EntityType]:
        return {entity.id: entity.type for entity in self.entities}

    def facts_for(self, relation: Relation) -> tuple[Fact, ...]:
        return tuple(fact for fact in self.facts if fact.relation == relation)


def _reachable(edges: Iterable[tuple[int, int]], start: int, target: int) -> bool:
    neighbors: dict[int, list[int]] = {}
    for left, right in edges:
        neighbors.setdefault(left, []).append(right)
    queue = deque([start])
    visited = {start}
    while queue:
        current = queue.popleft()
        for neighbor in sorted(neighbors.get(current, ())):
            if neighbor == target:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def entailed(world: World, query: Query) -> bool:
    """Return the complete relation-specific semantic target from REN-001C."""
    if query.relation == Relation.WITHIN:
        direct = {(f.arg1, f.arg2) for f in world.facts_for(Relation.DIRECT_IN)}
        if (query.arg1, query.arg2) in direct:
            return True
        nested = {(f.arg1, f.arg2) for f in world.facts_for(Relation.NESTED_IN)}
        return any(x == query.arg1 and (container, query.arg2) in nested for x, container in direct)
    if query.relation == Relation.BEFORE:
        edges = ((f.arg1, f.arg2) for f in world.facts_for(Relation.BEFORE))
        return _reachable(edges, query.arg1, query.arg2)
    if query.relation == Relation.SAME:
        if query.arg1 == query.arg2:
            return True
        edges = []
        for fact in world.facts_for(Relation.SAME):
            edges.extend(((fact.arg1, fact.arg2), (fact.arg2, fact.arg1)))
        return _reachable(edges, query.arg1, query.arg2)
    if query.relation == Relation.LINK:
        pair = tuple(sorted((query.arg1, query.arg2)))
        return any(tuple(sorted((f.arg1, f.arg2))) == pair for f in world.facts_for(Relation.LINK))
    raise ValueError(f"unsupported query relation: {query.relation.name}")


def validate_world(world: World) -> None:
    types = world.entity_types
    if len(types) != len(world.entities):
        raise ValueError("entity IDs must be unique")
    for fact in world.facts:
        expected = FACT_SIGNATURES.get(fact.relation)
        if expected is None:
            raise ValueError(f"query-only relation serialized as fact: {fact.relation.name}")
        actual = (types.get(fact.arg1), types.get(fact.arg2))
        if actual != expected:
            raise ValueError(f"ill-typed fact {fact}: expected {expected}, got {actual}")
    before = [(f.arg1, f.arg2) for f in world.facts_for(Relation.BEFORE)]
    nested = [(f.arg1, f.arg2) for f in world.facts_for(Relation.NESTED_IN)]
    for entity in types:
        if _reachable(before, entity, entity):
            raise ValueError("BEFORE must be acyclic")
        if _reachable(nested, entity, entity):
            raise ValueError("NESTED_IN must be acyclic")
    links = [tuple(sorted((f.arg1, f.arg2))) for f in world.facts_for(Relation.LINK)]
    if any(a == b for a, b in links) or len(links) != len(set(links)):
        raise ValueError("LINK must contain unique non-self undirected edges")


def validate_query(world: World, query: Query) -> None:
    expected = QUERY_SIGNATURES.get(query.relation)
    if expected is None:
        raise ValueError(f"fact-only relation used as query: {query.relation.name}")
    types = world.entity_types
    actual = (types.get(query.arg1), types.get(query.arg2))
    if actual != expected:
        raise ValueError(f"ill-typed query {query}: expected {expected}, got {actual}")
