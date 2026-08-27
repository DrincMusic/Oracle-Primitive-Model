from rlmgraph.opm.algebra import Entity, EntityType, Fact, Query, Relation, World, entailed


def _world() -> World:
    entities = tuple(
        [Entity(i, EntityType.OBJECT) for i in range(4)]
        + [Entity(i, EntityType.CONTAINER) for i in range(4, 7)]
    )
    facts = (
        Fact(Relation.DIRECT_IN, 0, 4),
        Fact(Relation.NESTED_IN, 4, 5),
        Fact(Relation.BEFORE, 0, 1),
        Fact(Relation.BEFORE, 1, 2),
        Fact(Relation.BEFORE, 2, 3),
        Fact(Relation.SAME, 0, 1),
        Fact(Relation.SAME, 1, 2),
        Fact(Relation.LINK, 2, 3),
    )
    return World(7, entities, facts)


def test_relation_specific_full_semantics() -> None:
    world = _world()
    assert entailed(world, Query(Relation.WITHIN, 0, 4))
    assert entailed(world, Query(Relation.WITHIN, 0, 5))
    assert not entailed(world, Query(Relation.WITHIN, 0, 6))
    assert entailed(world, Query(Relation.BEFORE, 0, 3))
    assert entailed(world, Query(Relation.SAME, 0, 2))
    assert entailed(world, Query(Relation.SAME, 2, 0))
    assert entailed(world, Query(Relation.LINK, 3, 2))
    assert not entailed(world, Query(Relation.LINK, 0, 3))
