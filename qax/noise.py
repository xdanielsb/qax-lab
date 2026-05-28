"""Simple single-qubit noise channels via Monte Carlo trajectories.

We support three textbook channels: bit-flip, phase-flip, and depolarizing.
Each is implemented as a randomized substitution: with probability ``p`` we
inject the matching Pauli onto the target wire; otherwise we leave the
state alone. Run the same circuit many times under :func:`jax.vmap` over a
batch of PRNG keys to get the mixed-state average.

This module deliberately keeps the channels *outside* the main statevector
backend so the noiseless simulator stays pure and easy to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from . import gates
from .simulate import _apply_gate
from .typing import DEFAULT_COMPLEX, Array, PRNGKey

Channel = Literal["bit_flip", "phase_flip", "depolarizing"]


@dataclass(frozen=True)
class NoiseChannel:
    """Apply one of the three textbook single-qubit channels at ``probability``."""

    kind: Channel
    wire: int
    probability: float


def apply_channel(
    state: Array,
    n_qubits: int,
    channel: NoiseChannel,
    key: PRNGKey,
    dtype=DEFAULT_COMPLEX,
) -> Array:
    """Sample a single Monte Carlo step for ``channel`` and return the new state."""
    if channel.kind == "bit_flip":
        u = jax.random.uniform(key)
        new_state = _apply_gate(state, gates.x(dtype=dtype), (channel.wire,), n_qubits)
        return jnp.where(u < channel.probability, new_state, state)

    if channel.kind == "phase_flip":
        u = jax.random.uniform(key)
        new_state = _apply_gate(state, gates.z(dtype=dtype), (channel.wire,), n_qubits)
        return jnp.where(u < channel.probability, new_state, state)

    if channel.kind == "depolarizing":
        # Split the key so the "should I flip" and "which Pauli" draws are independent.
        k_apply, k_which = jax.random.split(key)
        u = jax.random.uniform(k_apply)
        which = jax.random.randint(k_which, shape=(), minval=0, maxval=3)
        x_state = _apply_gate(state, gates.x(dtype=dtype), (channel.wire,), n_qubits)
        y_state = _apply_gate(state, gates.y(dtype=dtype), (channel.wire,), n_qubits)
        z_state = _apply_gate(state, gates.z(dtype=dtype), (channel.wire,), n_qubits)
        flipped = jnp.where(
            which == 0,
            x_state,
            jnp.where(which == 1, y_state, z_state),
        )
        return jnp.where(u < channel.probability, flipped, state)

    raise ValueError(f"Unknown noise channel kind: {channel.kind!r}")
