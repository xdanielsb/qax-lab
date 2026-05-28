"""Statevector helpers.

Statevectors are dense JAX arrays of shape ``(2**n,)`` in the computational
basis. The leftmost qubit is the most significant bit of the basis-state
index, so ``zero_state(3)[0] == 1`` corresponds to ``|000>`` and index
``5 = 0b101`` corresponds to ``|101>`` with qubit 0 = 1, qubit 1 = 0, qubit
2 = 1.
"""

from __future__ import annotations

import jax.numpy as jnp

from .typing import DEFAULT_COMPLEX, Array


def zero_state(n_qubits: int, dtype=DEFAULT_COMPLEX) -> Array:
    """Return the ``|0...0>`` statevector for ``n_qubits`` qubits."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1.")
    dim = 1 << n_qubits
    state = jnp.zeros(dim, dtype=dtype)
    return state.at[0].set(1)


def basis_state(n_qubits: int, index: int, dtype=DEFAULT_COMPLEX) -> Array:
    """Return ``|index>`` as a statevector."""
    dim = 1 << n_qubits
    if not 0 <= index < dim:
        raise ValueError(f"index {index} out of range for {n_qubits} qubits.")
    state = jnp.zeros(dim, dtype=dtype)
    return state.at[index].set(1)


def normalize(state: Array) -> Array:
    norm = jnp.sqrt(jnp.sum(jnp.abs(state) ** 2))
    return state / norm


def probabilities(state: Array) -> Array:
    """Return real-valued probabilities ``|amplitude|^2``."""
    return (state.conj() * state).real


def bitstring(index: int, n_qubits: int) -> str:
    """Render basis-state ``index`` as a left-MSB bitstring of length ``n_qubits``."""
    return format(index, f"0{n_qubits}b")
