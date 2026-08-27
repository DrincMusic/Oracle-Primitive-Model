from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn


@dataclass(frozen=True)
class ProbeResult:
    accuracy: float
    p_value: float
    rejects_null: bool
    passes_accuracy: bool


@dataclass(frozen=True)
class NeuralProbeResult:
    model_condition: str
    model_seed: int
    step: int
    accuracy: float
    p_value: float
    rejects_null: bool
    passes_accuracy: bool


@dataclass(frozen=True)
class RawLeakageProbeResult:
    channel: str
    accuracy: float
    correct: int
    count: int
    wilson_low: float
    wilson_high: float
    wilson_contains_half: bool
    p_value: float
    rejects_null: bool
    passes_accuracy: bool
    passes_descriptive_gate: bool


def wilson_interval(correct: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count <= 0:
        raise ValueError("count must be positive")
    proportion = correct / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = z * sqrt(proportion * (1 - proportion) / count + z * z / (4 * count**2))
    radius /= denominator
    return center - radius, center + radius


def categorical_leakage_probe(
    train_features: list[list[str]],
    train_labels: np.ndarray,
    validation_features: list[list[str]],
    validation_labels: np.ndarray,
    *,
    channel: str,
    threshold: float = 0.525,
) -> RawLeakageProbeResult:
    predictions = categorical_predictions(
        train_features, train_labels, validation_features
    )
    correct = int(np.sum(predictions == validation_labels))
    count = len(validation_labels)
    accuracy = correct / count
    low, high = wilson_interval(correct, count)
    p_value = float(binomtest(correct, count, p=0.5, alternative="greater").pvalue)
    contains_half = low <= 0.5 <= high
    return RawLeakageProbeResult(
        channel=channel,
        accuracy=accuracy,
        correct=correct,
        count=count,
        wilson_low=low,
        wilson_high=high,
        wilson_contains_half=contains_half,
        p_value=p_value,
        rejects_null=p_value < 0.05,
        passes_accuracy=accuracy <= threshold,
        passes_descriptive_gate=accuracy <= threshold and contains_half,
    )


def categorical_predictions(
    train_features: list[list[str]],
    train_labels: np.ndarray,
    validation_features: list[list[str]],
) -> np.ndarray:
    """Fit the frozen categorical probe and return validation predictions."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    train_encoded = encoder.fit_transform(train_features)
    validation_encoded = encoder.transform(validation_features)
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            fit_intercept=True,
            max_iter=2000,
            tol=1e-8,
            class_weight=None,
            random_state=91001,
        ),
    )
    classifier.fit(train_encoded, train_labels)
    return classifier.predict(validation_encoded)


def raw_probe_family_passes(results: list[RawLeakageProbeResult], alpha: float = 0.05) -> bool:
    if len(results) != 3 or len({result.channel for result in results}) != 3:
        raise ValueError("exactly three distinct raw oracle channels are required")
    ordered = sorted(results, key=lambda result: result.p_value)
    any_rejection = False
    for index, result in enumerate(ordered):
        if result.p_value < alpha / (3 - index):
            any_rejection = True
        else:
            break
    return not any_rejection and all(result.passes_descriptive_gate for result in results)


def logistic_leakage_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    threshold: float = 0.525,
) -> ProbeResult:
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            fit_intercept=True,
            max_iter=2000,
            tol=1e-8,
            class_weight=None,
            random_state=91001,
        ),
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(validation_features)
    correct = int(np.sum(predictions == validation_labels))
    count = len(validation_labels)
    accuracy = correct / count
    p_value = float(binomtest(correct, count, p=0.5, alternative="greater").pvalue)
    return ProbeResult(accuracy, p_value, p_value < 0.05, accuracy <= threshold)


def holm_fail_to_reject(results: list[ProbeResult], alpha: float = 0.05) -> bool:
    ordered = sorted(results, key=lambda result: result.p_value)
    count = len(ordered)
    for index, result in enumerate(ordered):
        if result.p_value >= alpha / (count - index):
            return all(item.passes_accuracy for item in results)
        return False
    return all(item.passes_accuracy for item in results)


def neural_evidence_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    model_condition: str,
    model_seed: int,
    step: int,
    epochs: int = 100,
    threshold: float = 0.55,
) -> NeuralProbeResult:
    if step not in (1, 2):
        raise ValueError("step must be 1 or 2")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    import hashlib

    material = f"opm-v1/neural-probe/{model_condition}/{model_seed}/{step}"
    seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    torch.manual_seed(seed % (2**63 - 1))
    mean = train_features.mean(axis=0, keepdims=True)
    scale = train_features.std(axis=0, keepdims=True)
    scale[scale == 0] = 1
    train_x = torch.tensor((train_features - mean) / scale, dtype=torch.float32)
    valid_x = torch.tensor((validation_features - mean) / scale, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    probe = nn.Sequential(nn.Linear(train_x.shape[1], 128), nn.GELU(), nn.Linear(128, 2))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed % (2**63 - 1))
    for _ in range(epochs):
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), 256):
            indices = order[start : start + 256]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(probe(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        predictions = probe(valid_x).argmax(dim=-1).numpy()
    correct = int(np.sum(predictions == validation_labels))
    count = len(validation_labels)
    accuracy = correct / count
    p_value = float(binomtest(correct, count, p=0.5, alternative="greater").pvalue)
    return NeuralProbeResult(
        model_condition=model_condition,
        model_seed=model_seed,
        step=step,
        accuracy=accuracy,
        p_value=p_value,
        rejects_null=p_value < 0.05,
        passes_accuracy=accuracy <= threshold,
    )


def neural_probe_pair_passes(results: list[NeuralProbeResult], alpha: float = 0.05) -> bool:
    if sorted(result.step for result in results) != [1, 2]:
        raise ValueError("one result is required for each evidence step")
    ordered = sorted(results, key=lambda result: result.p_value)
    for index, result in enumerate(ordered):
        if result.p_value >= alpha / (2 - index):
            return all(item.passes_accuracy for item in results)
        return False
    return all(item.passes_accuracy for item in results)
