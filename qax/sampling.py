"""Measurement sampling with explicit JAX PRNG keys.

We split this into a JAX-friendly numeric path (:func:`sample_indices`) and
a Python display helper (:func:`counts_from_indices`). Splitting keeps the
hot path traceable by ``jit``/``vmap`` while still giving users a nice
``{"010": 12, ...}`` representation when they need to print.
"""

from __future__ import annotations

from collections import Counter
from functools import partial

import jax
import jax.numpy as jnp

from .state import probabilities
from .typing import Array, PRNGKey


@partial(jax.jit, static_argnames=("n_shots",))
def sample_indices(key: PRNGKey, state: Array, n_shots: int) -> Array:
    """Sample basis-state integer indices according to ``|state|^2``.

    Returns an integer array of shape ``(n_shots,)``.
    """
    probs = probabilities(state)
    # ``jax.random.categorical`` works on logits; use log-probs with a tiny
    # epsilon to avoid -inf for impossible outcomes (zero probability).
    logits = jnp.log(probs + 1e-30)
    return jax.random.categorical(key, logits, shape=(n_shots,))


def counts_from_indices(indices: Array, n_qubits: int) -> dict[str, int]:
    """Group sampled integer indices into a ``{"bitstring": count}`` dict."""
    py_indices = [int(i) for i in jax.device_get(indices).tolist()]
    counter = Counter(py_indices)
    return {format(idx, f"0{n_qubits}b"): cnt for idx, cnt in sorted(counter.items())}


def sample_counts(key: PRNGKey, state: Array, n_qubits: int, n_shots: int) -> dict[str, int]:
    """Convenience wrapper: sample shots and return the count dictionary."""
    idx = sample_indices(key, state, n_shots)
    return counts_from_indices(idx, n_qubits)


def expected_counts(state: Array, n_qubits: int, n_shots: int) -> dict[str, float]:
    """Analytical expected counts for ``state`` — useful for tests/visuals."""
    probs = jax.device_get(probabilities(state))
    return {format(i, f"0{n_qubits}b"): float(p) * n_shots for i, p in enumerate(probs)}
