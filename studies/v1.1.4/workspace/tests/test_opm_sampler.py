from collections import Counter, defaultdict

from rlmgraph.opm.generation import Operation
from rlmgraph.opm.sampler import CanonicalSampler, batch_quotas
from rlmgraph.opm.splits import HOLDOUTS


def _pools(seed: int, size: int = 40):
    return {
        stratum: tuple(f"{int(stratum[0])}-{int(stratum[1])}-{stratum[2]}-{i}" for i in range(size))
        for stratum in batch_quotas(seed, 0)
    }


def test_complete_run_marginals_and_terminal_tail() -> None:
    for seed in (1101, 2202, 3303, 4404, 5505):
        total: Counter = Counter()
        for step in range(50_000):
            quotas = batch_quotas(seed, step)
            assert sum(quotas.values()) == 256
            assert all(operation != HOLDOUTS[domain] for domain, operation, _ in quotas)
            total.update(quotas)
        labels = Counter()
        operations = Counter()
        domains = Counter()
        steps = Counter()
        for (domain, operation, label), count in total.items():
            labels[label] += count
            operations[operation] += count
            domains[domain] += count
            steps[1 if operation in (Operation.LOOKUP, Operation.REVERSE) else 2] += count
        assert set(labels.values()) == {6_400_000}
        assert set(operations.values()) == {3_200_000}
        assert set(steps.values()) == {6_400_000}
        assert max(domains.values()) - min(domains.values()) == 1
        assert batch_quotas(seed, 49_998) == batch_quotas(seed, 0)
        assert batch_quotas(seed, 49_999) == batch_quotas(seed, 1)


def test_no_replacement_replay_and_resume() -> None:
    seed = 1101
    first = CanonicalSampler(_pools(seed), seed)
    seen = defaultdict(set)
    batches = []
    for _ in range(3):
        batch = first.next_batch()
        batches.append(batch)
        for item in batch:
            key = (item.stratum, item.reuse_cycle)
            assert item.example_id not in seen[key]
            seen[key].add(item.example_id)

    replay = CanonicalSampler(_pools(seed), seed)
    assert [replay.next_batch() for _ in range(3)] == batches

    state = first.state_dict()
    expected = first.next_batch()
    resumed = CanonicalSampler(_pools(seed), seed)
    resumed.load_state_dict(state)
    assert resumed.next_batch() == expected
