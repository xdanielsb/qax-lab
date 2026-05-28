"""Tests for gate matrices."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from qax import gates
from qax.gates import gate_matrix

SINGLE_QUBIT_NAMES = ["i", "x", "y", "z", "h", "s", "sdg", "t", "tdg"]
TWO_QUBIT_NAMES = ["cnot", "cz", "swap"]


def _is_unitary(u: jnp.ndarray, atol: float = 1e-5) -> bool:
    n = u.shape[0]
    return np.allclose(np.asarray(u.conj().T @ u), np.eye(n), atol=atol)


@pytest.mark.parametrize("name", SINGLE_QUBIT_NAMES)
def test_single_qubit_shape_and_unitary(name: str) -> None:
    u = gate_matrix(name)
    assert u.shape == (2, 2)
    assert _is_unitary(u)


@pytest.mark.parametrize("name", TWO_QUBIT_NAMES)
def test_two_qubit_shape_and_unitary(name: str) -> None:
    u = gate_matrix(name)
    assert u.shape == (4, 4)
    assert _is_unitary(u)


@pytest.mark.parametrize("name", ["rx", "ry", "rz"])
@pytest.mark.parametrize("theta", [0.0, 0.1, np.pi / 3, -1.7, 2 * np.pi])
def test_parameterized_unitary(name: str, theta: float) -> None:
    u = gate_matrix(name, theta)
    assert u.shape == (2, 2)
    assert _is_unitary(u)


def test_rx_zero_is_identity() -> None:
    np.testing.assert_allclose(np.asarray(gates.rx(0.0)), np.eye(2), atol=1e-6)


def test_ry_zero_is_identity() -> None:
    np.testing.assert_allclose(np.asarray(gates.ry(0.0)), np.eye(2), atol=1e-6)


def test_rz_zero_is_identity() -> None:
    np.testing.assert_allclose(np.asarray(gates.rz(0.0)), np.eye(2), atol=1e-6)


def test_rz_pi_is_minus_iz() -> None:
    expected = np.array([[np.exp(-1j * np.pi / 2), 0], [0, np.exp(1j * np.pi / 2)]])
    np.testing.assert_allclose(np.asarray(gates.rz(np.pi)), expected, atol=1e-6)


def test_unknown_gate_raises() -> None:
    with pytest.raises(ValueError):
        gate_matrix("foo")


def test_param_required_for_parameterized() -> None:
    with pytest.raises(ValueError):
        gate_matrix("rx")


def test_h_on_zero_is_plus() -> None:
    plus = gates.h() @ jnp.array([1.0, 0.0], dtype=jnp.complex64)
    np.testing.assert_allclose(np.asarray(plus), np.array([1, 1]) / np.sqrt(2), atol=1e-6)


def test_cnot_matrix_explicit() -> None:
    expected = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64
    )
    np.testing.assert_allclose(np.asarray(gates.cnot()), expected)
