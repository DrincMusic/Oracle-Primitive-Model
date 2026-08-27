from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .algebra import EntityType, Fact, Query, Relation, World
from .generation import derive_uint64


class Domain(IntEnum):
    SET = 0
    SCENE = 1
    PROGRAM = 2


SET_ALIASES = {
    Relation.DIRECT_IN: "member",
    Relation.NESTED_IN: "subset",
    Relation.BEFORE: "ordered_before",
    Relation.SAME: "equivalent",
    Relation.LINK: "paired",
    Relation.WITHIN: "within",
}
SCENE_ALIASES = {
    Relation.DIRECT_IN: "@",
    Relation.NESTED_IN: "<<",
    Relation.BEFORE: "<t",
    Relation.SAME: "==",
    Relation.LINK: "--",
    Relation.WITHIN: "WITHIN",
}
PROGRAM_ALIASES = {
    Relation.DIRECT_IN: "has",
    Relation.NESTED_IN: "imports",
    Relation.BEFORE: "precedes",
    Relation.SAME: "alias",
    Relation.LINK: "connected",
    Relation.WITHIN: "within",
}
V2_ALIASES = {
    Domain.SET: {
        Relation.DIRECT_IN: "paired",
        Relation.NESTED_IN: "member",
        Relation.BEFORE: "subset",
        Relation.SAME: "ordered_before",
        Relation.LINK: "equivalent",
        Relation.WITHIN: "within",
    },
    Domain.SCENE: {
        Relation.DIRECT_IN: "--",
        Relation.NESTED_IN: "@",
        Relation.BEFORE: "<<",
        Relation.SAME: "<t",
        Relation.LINK: "==",
        Relation.WITHIN: "WITHIN",
    },
    Domain.PROGRAM: {
        Relation.DIRECT_IN: "connected",
        Relation.NESTED_IN: "has",
        Relation.BEFORE: "imports",
        Relation.SAME: "precedes",
        Relation.LINK: "alias",
        Relation.WITHIN: "within",
    },
}


@dataclass(frozen=True)
class RenderedRecord:
    facts: tuple[str, ...]
    query: str
    surface_to_entity: dict[str, int]


def _inventories(domain: Domain) -> tuple[list[str], list[str]]:
    if domain == Domain.SET:
        return ([f"e{i:02d}" for i in range(32)], [f"s{i:02d}" for i in range(16)])
    if domain == Domain.SCENE:
        return ([f"o{i:02d}" for i in range(32)], [f"r{i:02d}" for i in range(16)])
    return ([f"v{i:02d}" for i in range(32)], [f"c{i:02d}" for i in range(16)])


def entity_names(world: World, domain: Domain, variant: int) -> dict[int, str]:
    objects, containers = _inventories(domain)
    rng = np.random.Generator(
        np.random.PCG64DXSM(derive_uint64(world.world_id, "rename", domain.name))
    )
    objects = [objects[int(i)] for i in rng.permutation(len(objects))]
    containers = [containers[int(i)] for i in rng.permutation(len(containers))]
    if variant == 2:
        object_rotation = derive_uint64("opm-v1", "renderer-v2", world.world_id, domain.name) % len(objects)
        container_rotation = derive_uint64(
            "opm-v1", "renderer-v2", world.world_id, domain.name, "container"
        ) % len(containers)
        objects = objects[object_rotation:] + objects[:object_rotation]
        containers = containers[container_rotation:] + containers[:container_rotation]
    object_ids = sorted(entity.id for entity in world.entities if entity.type == EntityType.OBJECT)
    container_ids = sorted(
        entity.id for entity in world.entities if entity.type == EntityType.CONTAINER
    )
    return {
        **{entity_id: objects[index] for index, entity_id in enumerate(object_ids)},
        **{entity_id: containers[index] for index, entity_id in enumerate(container_ids)},
    }


def _alias(domain: Domain, relation: Relation, variant: int) -> str:
    if variant == 2:
        return V2_ALIASES[domain][relation]
    return {Domain.SET: SET_ALIASES, Domain.SCENE: SCENE_ALIASES, Domain.PROGRAM: PROGRAM_ALIASES}[
        domain
    ][relation]


def _args(domain: Domain, relation: Relation, left: str, right: str) -> tuple[str, str]:
    if domain == Domain.PROGRAM and relation in (Relation.DIRECT_IN, Relation.NESTED_IN):
        return right, left
    return left, right


def render_fact(fact: Fact, names: dict[int, str], domain: Domain, variant: int) -> str:
    left, right = _args(domain, fact.relation, names[fact.arg1], names[fact.arg2])
    alias = _alias(domain, fact.relation, variant)
    if variant == 2:
        return f"[ {alias} , {left} , {right} ]"
    if domain == Domain.SCENE:
        return f"{left} {alias} {right}" if variant == 0 else f"[ {left} , {alias} , {right} ]"
    if variant == 1:
        return f"( {left} , {right} ) : {alias}"
    return f"{alias} ( {left} , {right} )"


def render_query(query: Query, names: dict[int, str], domain: Domain, variant: int) -> str:
    left, right = _args(domain, query.relation, names[query.arg1], names[query.arg2])
    alias = _alias(domain, query.relation, variant)
    if variant == 2:
        return f"[ ? , {alias} , {left} , {right} ]"
    if domain == Domain.SET:
        return f"ask : {alias} ( {left} , {right} )"
    if domain == Domain.SCENE:
        return f"? {alias} {left} {right}"
    return f"assert ? {alias} ( {left} , {right} )"


def render(world: World, facts: Iterable[Fact], query: Query, domain: Domain, variant: int) -> RenderedRecord:
    if variant not in (0, 1, 2):
        raise ValueError("renderer variant must be 0, 1, or 2")
    names = entity_names(world, domain, variant)
    rendered_facts = tuple(render_fact(fact, names, domain, variant) for fact in facts)
    if variant == 2:
        rendered_facts = tuple(reversed(rendered_facts))
    return RenderedRecord(
        facts=rendered_facts,
        query=render_query(query, names, domain, variant),
        surface_to_entity={surface: entity_id for entity_id, surface in names.items()},
    )


TOKEN_RE = re.compile(r"<<|<t|==|--|[()\[\],;:?@]|[A-Za-z][A-Za-z0-9_]*")


def lex(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor].isspace():
            cursor += 1
            continue
        match = TOKEN_RE.match(text, cursor)
        if match is None:
            raise ValueError(f"unmatched lexer input at offset {cursor}: {text[cursor:cursor+12]!r}")
        tokens.append(match.group(0))
        cursor = match.end()
    return tuple(tokens)


def fact_order_key(example_id: str, fact: Fact, occurrence: int) -> str:
    material = f"fact-order/{example_id}/{fact.relation.name}/{fact.arg1}/{fact.arg2}/{occurrence}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
