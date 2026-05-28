"""Tests for measurement sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qax import (
    Circuit,
    counts_from_indices,
    probabilities,
    sample_counts,
    sample_indices,
    simulate,
    zero_state,
)


def test_sample_returns_only_zero_for_zero_state() -> None:
    key = jax.random.PRNGKey(0)
    idx = sample_indices(key, zero_state(3), n_shots=256)
    assert jnp.all(idx == 0)


def test_sample_bell_distribution_close_to_half_half() -> None:
    key = jax.random.PRNGKey(0)
    state = simulate(Circuit(2).h(0).cx(0, 1))
    n = 50_000
    idx = sample_indices(key, state, n_shots=n)
    counts = np.bincount(np.asarray(idx), minlength=4)
    # |00> = index 0, |11> = index 3. Others should be ~0.
    assert counts[1] == 0 and counts[2] == 0
    frac00 = counts[0] / n
    assert 0.45 < frac00 < 0.55


def test_counts_from_indices_returns_correct_keys() -> None:
    idx = jnp.array([0, 3, 0, 3, 0])
    counts = counts_from_indices(idx, n_qubits=2)
    assert counts == {"00": 3, "11": 2}


def test_sample_frequencies_match_probabilities() -> None:
    key = jax.random.PRNGKey(42)
    state = simulate(Circuit(3).h(0).ry(1, "a").cx(0, 2), params={"a": jnp.array(0.6)})
    n = 100_000
    counts = sample_counts(key, state, n_qubits=3, n_shots=n)
    probs = np.asarray(probabilities(state))
    for idx, prob in enumerate(probs):
        bs = format(idx, "03b")
        observed = counts.get(bs, 0) / n
        # 100k shots → 1/sqrt(n) ≈ 0.003 stderr; 0.01 tolerance is comfortable.
        assert abs(observed - prob) < 0.01


def test_jit_sample_path() -> None:
    key = jax.random.PRNGKey(1)
    state = simulate(Circuit(2).h(0).cx(0, 1))
    # ``sample_indices`` is already jitted; calling it twice with the same
    # static n_shots should hit the cache and return shape-matching results.
    a = sample_indices(key, state, n_shots=8)
    b = sample_indices(key, state, n_shots=8)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
