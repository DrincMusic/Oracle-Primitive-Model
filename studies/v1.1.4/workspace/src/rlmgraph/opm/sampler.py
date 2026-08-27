from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from .generation import Operation, derive_uint64
from .rendering import Domain
from .splits import HOLDOUTS

Stratum = tuple[Domain, Operation, int]


def _rotate(values: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    amount %= 3
    return values[-amount:] + values[:-amount] if amount else values


def batch_quotas(seed: int, step: int) -> Counter[Stratum]:
    """Return the approved v1.1.4 quota table for one zero-based optimizer step."""
    if step < 0 or step >= 50_000:
        raise ValueError("canonical step must be in 0..49999")
    batch = step % 3
    rotation = derive_uint64("opm-v1.1.4", "batch-partition", seed) % 3
    label_lookup = {
        0: _rotate((11, 11, 10), rotation + batch),
        1: _rotate((11, 10, 11), rotation + batch),
    }
    quotas: Counter[Stratum] = Counter()
    for label in (0, 1):
        for domain in Domain:
            quotas[(domain, Operation.LOOKUP, label)] = label_lookup[label][int(domain)]
            for operation in Operation:
                if operation != Operation.LOOKUP and operation != HOLDOUTS[domain]:
                    quotas[(domain, operation, label)] = 16
    if sum(quotas.values()) != 256:
        raise AssertionError("v1.1.4 batch quota does not sum to 256")
    return quotas


@dataclass(frozen=True)
class SampleSelection:
    example_id: str
    stratum: Stratum
    reuse_cycle: int


class CanonicalSampler:
    """Stateful deterministic selection without replacement inside each reuse cycle."""

    def __init__(self, pools: dict[Stratum, tuple[str, ...]], seed: int) -> None:
        expected = set(batch_quotas(seed, 0))
        if set(pools) != expected or any(not pool for pool in pools.values()):
            raise ValueError("sampler pools must cover every supported nonempty stratum")
        self.pools = {key: tuple(sorted(pool)) for key, pool in pools.items()}
        if any(len(set(pool)) != len(pool) for pool in self.pools.values()):
            raise ValueError("sampler pools contain duplicate example IDs")
        self.seed = seed
        self.next_step = 0
        self.cycles = {key: 0 for key in self.pools}
        self.used = {key: set() for key in self.pools}

    def _take(self, stratum: Stratum, count: int, macrocycle: int) -> list[SampleSelection]:
        selected: list[SampleSelection] = []
        while len(selected) < count:
            cycle = self.cycles[stratum]
            remaining = set(self.pools[stratum]) - self.used[stratum]
            if not remaining:
                self.cycles[stratum] += 1
                self.used[stratum].clear()
                continue
            ranked = sorted(
                remaining,
                key=lambda example_id: hashlib.sha256(
                    (
                        f"opm-v1.1.4/sample/{self.seed}/{macrocycle}/"
                        f"{int(stratum[0])},{int(stratum[1])},{stratum[2]}/{cycle}/{example_id}"
                    ).encode()
                ).hexdigest(),
            )
            take = ranked[: count - len(selected)]
            self.used[stratum].update(take)
            selected.extend(SampleSelection(item, stratum, cycle) for item in take)
        return selected

    def next_batch(self) -> tuple[SampleSelection, ...]:
        if self.next_step >= 50_000:
            raise StopIteration("canonical sampler is exhausted")
        step = self.next_step
        macrocycle = step // 3
        selected: list[SampleSelection] = []
        for stratum, count in sorted(batch_quotas(self.seed, step).items()):
            selected.extend(self._take(stratum, count, macrocycle))
        selected.sort(
            key=lambda item: hashlib.sha256(
                f"opm-v1.1.4/batch-order/{self.seed}/{step}/{item.example_id}".encode()
            ).hexdigest()
        )
        self.next_step += 1
        return tuple(selected)

    def state_dict(self) -> dict[str, object]:
        key = lambda value: f"{int(value[0])},{int(value[1])},{value[2]}"
        return {
            "seed": self.seed,
            "next_step": self.next_step,
            "cycles": {key(item): cycle for item, cycle in self.cycles.items()},
            "used": {key(item): sorted(values) for item, values in self.used.items()},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("sampler-state seed mismatch")
        encoded_to_key = {
            f"{int(item[0])},{int(item[1])},{item[2]}": item for item in self.pools
        }
        cycles = dict(state["cycles"])
        used = dict(state["used"])
        if set(cycles) != set(encoded_to_key) or set(used) != set(encoded_to_key):
            raise ValueError("sampler-state stratum mismatch")
        self.next_step = int(state["next_step"])
        self.cycles = {encoded_to_key[item]: int(value) for item, value in cycles.items()}
        self.used = {encoded_to_key[item]: set(values) for item, values in used.items()}
        for stratum, values in self.used.items():
            if not values <= set(self.pools[stratum]):
                raise ValueError("sampler state references an unknown example")
