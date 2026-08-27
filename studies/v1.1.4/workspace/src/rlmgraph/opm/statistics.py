from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedPrediction:
    model_seed: int
    example_key: str
    shared_correct: int
    generalist_correct: int
    untied_correct: int


@dataclass(frozen=True)
class BootstrapResult:
    delta_generalist: float
    delta_untied: float
    generalist_interval: tuple[float, float]
    untied_interval: tuple[float, float]
    replicates: int


def paired_two_level_bootstrap(
    predictions: Iterable[PairedPrediction], *, replicates: int = 10_000, seed: int = 99117
) -> BootstrapResult:
    rows = list(predictions)
    if not rows:
        raise ValueError("bootstrap requires predictions")
    by_seed: dict[int, list[PairedPrediction]] = {}
    for row in rows:
        by_seed.setdefault(row.model_seed, []).append(row)
    seeds = sorted(by_seed)
    if any(len({item.example_key for item in group}) != len(group) for group in by_seed.values()):
        raise ValueError("duplicate pairing key within model seed")
    key_sets = [{item.example_key for item in by_seed[item]} for item in seeds]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("model seeds must contain identical example keys")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    generalist_samples = np.empty(replicates, dtype=np.float64)
    untied_samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        generalist_differences: list[float] = []
        untied_differences: list[float] = []
        for sampled_seed in sampled_seeds:
            group = by_seed[int(sampled_seed)]
            indices = rng.integers(0, len(group), size=len(group))
            for index in indices:
                row = group[int(index)]
                generalist_differences.append(row.shared_correct - row.generalist_correct)
                untied_differences.append(row.shared_correct - row.untied_correct)
        generalist_samples[replicate] = float(np.mean(generalist_differences))
        untied_samples[replicate] = float(np.mean(untied_differences))
    point_generalist = float(
        np.mean([row.shared_correct - row.generalist_correct for row in rows])
    )
    point_untied = float(np.mean([row.shared_correct - row.untied_correct for row in rows]))
    return BootstrapResult(
        delta_generalist=point_generalist,
        delta_untied=point_untied,
        generalist_interval=tuple(np.quantile(generalist_samples, [0.025, 0.975]).tolist()),
        untied_interval=tuple(np.quantile(untied_samples, [0.025, 0.975]).tolist()),
        replicates=replicates,
    )
