from rlmgraph.opm.algebra import Query, Relation
from rlmgraph.opm.generation import generate_world
from rlmgraph.opm.rendering import Domain, entity_names, lex, render


def test_renderer_is_deterministic_and_cross_domain_names_differ() -> None:
    world = generate_world(222, n_objects=8, n_containers=4)
    fact = world.facts[0]
    query_relation = Relation.WITHIN if fact.relation == Relation.DIRECT_IN else fact.relation
    query = Query(query_relation, fact.arg1, fact.arg2)
    set_record = render(world, [fact], query, Domain.SET, 0)
    assert set_record == render(world, [fact], query, Domain.SET, 0)
    assert entity_names(world, Domain.SET, 0)[fact.arg1] != entity_names(world, Domain.SCENE, 0)[
        fact.arg1
    ]


def test_variant_two_is_prefix_and_lexable() -> None:
    world = generate_world(333, n_objects=8, n_containers=4)
    fact = world.facts_for(Relation.LINK)[0]
    query = Query(Relation.LINK, fact.arg2, fact.arg1)
    record = render(world, [fact], query, Domain.PROGRAM, 2)
    assert record.facts[0].startswith("[")
    assert record.query.startswith("[ ?")
    assert lex(record.facts[0])
    assert lex(record.query)
