"""Tests for the Monte Carlo noise channels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qax import NoiseChannel, apply_channel, simulate, zero_state
from qax.circuit import Circuit


def test_bit_flip_at_p_zero_is_no_op() -> None:
    state = zero_state(1)
    key = jax.random.PRNGKey(0)
    out = apply_channel(state, 1, NoiseChannel("bit_flip", 0, 0.0), key)
    np.testing.assert_allclose(np.asarray(out), np.asarray(state), atol=1e-6)


def test_bit_flip_at_p_one_flips() -> None:
    state = zero_state(1)
    key = jax.random.PRNGKey(0)
    out = apply_channel(state, 1, NoiseChannel("bit_flip", 0, 1.0), key)
    # The deterministic outcome is X|0> = |1>.
    np.testing.assert_allclose(np.asarray(out), np.array([0, 1], dtype=np.complex64), atol=1e-6)


def test_phase_flip_on_plus_state_yields_minus() -> None:
    state = simulate(Circuit(1).h(0))
    key = jax.random.PRNGKey(0)
    out = apply_channel(state, 1, NoiseChannel("phase_flip", 0, 1.0), key)
    expected = jnp.array([1, -1], dtype=jnp.complex64) / jnp.sqrt(2)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), atol=1e-6)


def test_depolarizing_average_decays_z() -> None:
    # Under depolarizing noise at strength p on |0>, <Z> averages to (1 - 4p/3)
    # in the standard convention. Just check that the average decays in (0, 1).
    state = zero_state(1)
    p = 0.4

    def trajectory(key):
        return apply_channel(state, 1, NoiseChannel("depolarizing", 0, p), key)

    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    traj_states = jax.vmap(trajectory)(keys)
    z_per_traj = jnp.abs(traj_states[:, 0]) ** 2 - jnp.abs(traj_states[:, 1]) ** 2
    mean_z = float(jnp.mean(z_per_traj))
    assert 0.0 < mean_z < 1.0
