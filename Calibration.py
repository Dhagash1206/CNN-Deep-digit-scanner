"""Temperature scaling: one scalar T that makes softmax confidences match observed accuracy."""

import json
import os

import numpy as np

EPSILON = 1e-9


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    # log(softmax(z)) equals z up to a per-row constant, so it can stand in for the logits.
    logits = np.log(np.clip(probabilities, EPSILON, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    picked = probabilities[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(picked, EPSILON, 1.0))))


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
            error += gap * in_bin.mean()
    return float(error)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.geomspace(0.25, 8.0, 400)
    losses = [
        negative_log_likelihood(apply_temperature(probabilities, t), labels)
        for t in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def save_temperature(path: str, temperature: float) -> None:
    with open(path, "w") as file:
        json.dump({"temperature": temperature}, file, indent=2)


def load_temperature(path: str) -> float:
    if not os.path.exists(path):
        return 1.0
    with open(path) as file:
        return float(json.load(file)["temperature"])
