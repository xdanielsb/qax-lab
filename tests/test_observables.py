"""Tests for Pauli observables and expectation values."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from qax import (
    Circuit,
    PauliHamiltonian,
    apply_pauli_string,
    expectation,
    simulate,
    zero_state,
)


def test_z_expectation_on_zero_is_plus_one() -> None:
    state = zero_state(1)
    h = PauliHamiltonian([("Z", 1.0)])
    assert np.isclose(float(expectation(state, h)), 1.0, atol=1e-5)


def test_z_expectation_on_one_is_minus_one() -> None:
    state = simulate(Circuit(1).x(0))
    h = PauliHamiltonian([("Z", 1.0)])
    assert np.isclose(float(expectation(state, h)), -1.0, atol=1e-5)


def test_x_expectation_on_plus_is_plus_one() -> None:
    state = simulate(Circuit(1).h(0))
    h = PauliHamiltonian([("X", 1.0)])
    assert np.isclose(float(expectation(state, h)), 1.0, atol=1e-5)


def test_bell_state_zz_expectation() -> None:
    # Bell state |Phi+> satisfies <ZZ> = +1.
    state = simulate(Circuit(2).h(0).cx(0, 1))
    h = PauliHamiltonian([("ZZ", 1.0)])
    assert np.isclose(float(expectation(state, h)), 1.0, atol=1e-5)


def test_bell_state_xx_expectation() -> None:
    state = simulate(Circuit(2).h(0).cx(0, 1))
    h = PauliHamiltonian([("XX", 1.0)])
    assert np.isclose(float(expectation(state, h)), 1.0, atol=1e-5)


def test_linearity_of_expectation() -> None:
    state = simulate(Circuit(2).h(0).cx(0, 1))
    h1 = PauliHamiltonian([("ZZ", 1.0)])
    h2 = PauliHamiltonian([("XX", 1.0)])
    h12 = PauliHamiltonian([("ZZ", 1.0), ("XX", 1.0)])
    assert np.isclose(
        float(expectation(state, h12)),
        float(expectation(state, h1)) + float(expectation(state, h2)),
        atol=1e-5,
    )


def test_invalid_pauli_string_rejected() -> None:
    with pytest.raises(ValueError):
        PauliHamiltonian([("ZA", 1.0)])


def test_mismatched_widths_rejected() -> None:
    with pytest.raises(ValueError):
        PauliHamiltonian([("Z", 1.0), ("ZZ", 1.0)])


def test_apply_pauli_string_returns_correct_shape() -> None:
    state = zero_state(3)
    out = apply_pauli_string(state, "XYZ")
    assert out.shape == (8,)


def test_identity_pauli_is_no_op() -> None:
    state = simulate(Circuit(2).h(0).cx(0, 1))
    out = apply_pauli_string(state, "II")
    np.testing.assert_allclose(np.asarray(out), np.asarray(state), atol=1e-6)


def test_expectation_value_is_real() -> None:
    state = simulate(Circuit(2).h(0).cx(0, 1))
    h = PauliHamiltonian([("XY", 0.5), ("YX", 0.5)])
    val = expectation(state, h)
    assert val.dtype in (jnp.float32, jnp.float64)
