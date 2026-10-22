import numpy as np

import calibration as cal


def _softmax(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def test_temperature_one_is_identity():
    probs = _softmax(np.random.default_rng(0).normal(size=(5, 10)))
    np.testing.assert_allclose(cal.apply_temperature(probs, 1.0), probs, atol=1e-6)


def test_higher_temperature_softens():
    probs = _softmax(np.array([[4.0, 1.0, 0.0]]))
    assert cal.apply_temperature(probs, 2.0).max() < probs.max()


def test_fit_recovers_overconfidence():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=4000)
    true_logits = rng.normal(size=(4000, 10))
    true_logits[np.arange(4000), labels] += 2.0
    # Draw labels from the true distribution, then make the model 3x overconfident.
    true_probs = _softmax(true_logits)
    sampled = np.array([rng.choice(10, p=p) for p in true_probs])
    overconfident = _softmax(true_logits * 3.0)
    temperature = cal.fit_temperature(overconfident, sampled)
    assert 2.5 < temperature < 3.5


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "calibration.json"
    cal.save_temperature(str(path), 1.7)
    assert cal.load_temperature(str(path)) == 1.7
    assert cal.load_temperature(str(tmp_path / "missing.json")) == 1.0
