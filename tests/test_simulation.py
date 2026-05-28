"""Tests for the simulation backends (eager, jit, scan)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from qax import Circuit, simulate, simulate_scan, zero_state


def test_h_then_h_returns_zero() -> None:
    c = Circuit(1).h(0).h(0)
    state = simulate(c)
    np.testing.assert_allclose(np.asarray(state), np.asarray(zero_state(1)), atol=1e-5)


def test_h_on_zero_is_plus() -> None:
    c = Circuit(1).h(0)
    state = simulate(c)
    expected = np.array([1, 1]) / np.sqrt(2)
    np.testing.assert_allclose(np.asarray(state), expected, atol=1e-5)


def test_bell_state() -> None:
    c = Circuit(2).h(0).cx(0, 1)
    state = simulate(c)
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(np.asarray(state), expected, atol=1e-5)


def test_ghz_state() -> None:
    c = Circuit(3).h(0).cx(0, 1).cx(1, 2)
    state = simulate(c)
    expected = np.zeros(8, dtype=np.complex64)
    expected[0] = expected[7] = 1 / np.sqrt(2)
    np.testing.assert_allclose(np.asarray(state), expected, atol=1e-5)


def test_jit_and_eager_match() -> None:
    c = Circuit(3).h(0).rx(1, "a").cx(0, 1).ry(2, "b").cz(1, 2).rz(0, "c")
    params = {"a": jnp.array(0.4), "b": jnp.array(-1.1), "c": jnp.array(0.27)}
    s_jit = simulate(c, params, jit=True)
    s_eager = simulate(c, params, jit=False)
    np.testing.assert_allclose(np.asarray(s_jit), np.asarray(s_eager), atol=1e-5)


def test_scan_and_eager_match() -> None:
    c = Circuit(3).h(0).rx(1, "a").cx(0, 1).ry(2, "b").cz(1, 2).rz(0, "c")
    params = {"a": jnp.array(0.4), "b": jnp.array(-1.1), "c": jnp.array(0.27)}
    s_scan = simulate_scan(c, params)
    s_eager = simulate(c, params, jit=False)
    np.testing.assert_allclose(np.asarray(s_scan), np.asarray(s_eager), atol=1e-4)


def test_swap_gate() -> None:
    c = Circuit(2).x(0).swap(0, 1)
    state = simulate(c)
    # After X on q0 then swap: q0=0, q1=1 → |01> which has index 1.
    expected = np.zeros(4, dtype=np.complex64)
    expected[1] = 1.0
    np.testing.assert_allclose(np.asarray(state), expected, atol=1e-5)


def test_circuit_with_no_ops_returns_initial_state() -> None:
    c = Circuit(3)
    state = simulate(c)
    np.testing.assert_allclose(np.asarray(state), np.asarray(zero_state(3)), atol=1e-6)


def test_missing_param_raises() -> None:
    c = Circuit(1).rx(0, "theta")
    with pytest.raises(KeyError):
        simulate(c, params={})


def test_normalization_preserved() -> None:
    c = Circuit(4).h(0).h(1).cx(0, 2).ry(3, "a").cz(1, 3).rx(2, "b").cx(2, 3)
    state = simulate(c, params={"a": jnp.array(0.7), "b": jnp.array(-0.3)})
    assert np.isclose(float(jnp.sum(jnp.abs(state) ** 2)), 1.0, atol=1e-5)
