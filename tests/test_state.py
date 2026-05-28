"""Tests for state-vector helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from qax import basis_state, normalize, probabilities, zero_state
from qax.state import bitstring


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_zero_state_shape_and_norm(n: int) -> None:
    s = zero_state(n)
    assert s.shape == (2**n,)
    assert s.dtype == jnp.complex64
    assert np.isclose(float(jnp.sum(probabilities(s))), 1.0, atol=1e-6)
    assert s[0] == 1


def test_basis_state_indexes_correctly() -> None:
    s = basis_state(3, 5)
    assert s.shape == (8,)
    assert int(jnp.argmax(jnp.abs(s))) == 5


def test_basis_state_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        basis_state(2, 4)


def test_zero_state_rejects_zero_qubits() -> None:
    with pytest.raises(ValueError):
        zero_state(0)


def test_probabilities_sum_to_one() -> None:
    s = jnp.array([0.5, 0.5j, 0.5, -0.5j], dtype=jnp.complex64)
    assert np.isclose(float(jnp.sum(probabilities(s))), 1.0, atol=1e-6)


def test_normalize_idempotent() -> None:
    s = jnp.array([1.0, 1.0], dtype=jnp.complex64)
    n = normalize(s)
    n2 = normalize(n)
    np.testing.assert_allclose(np.asarray(n), np.asarray(n2), atol=1e-6)
    assert np.isclose(float(jnp.sum(probabilities(n))), 1.0, atol=1e-6)


def test_bitstring_format() -> None:
    assert bitstring(0, 3) == "000"
    assert bitstring(5, 3) == "101"
    assert bitstring(7, 4) == "0111"
