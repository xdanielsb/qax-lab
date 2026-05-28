"""Tests that gradients flow through the simulator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qax import Circuit, PauliHamiltonian, expectation, simulate
from qax.grad_utils import parameter_shift_grad


def _energy(params, circuit, hamiltonian):
    return expectation(simulate(circuit, params), hamiltonian)


def test_grad_is_finite() -> None:
    circuit = Circuit(2).ry(0, "a").cx(0, 1).rz(1, "b")
    hamiltonian = PauliHamiltonian([("ZZ", 1.0)])
    params = {"a": jnp.array(0.3), "b": jnp.array(-0.7)}

    grads = jax.grad(_energy)(params, circuit, hamiltonian)
    for g in jax.tree.leaves(grads):
        assert jnp.isfinite(g).all()


def test_grad_zero_at_minimum() -> None:
    # <Z> on RY(0, theta)|0>: cos(theta). Derivative is -sin(theta) = 0 at theta=0.
    circuit = Circuit(1).ry(0, "a")
    hamiltonian = PauliHamiltonian([("Z", 1.0)])
    params = {"a": jnp.array(0.0)}
    grad_val = jax.grad(_energy)(params, circuit, hamiltonian)["a"]
    assert abs(float(grad_val)) < 1e-5


def test_value_and_grad_pair() -> None:
    circuit = Circuit(1).ry(0, "a")
    hamiltonian = PauliHamiltonian([("Z", 1.0)])
    params = {"a": jnp.array(1.234)}
    val, grad = jax.value_and_grad(_energy)(params, circuit, hamiltonian)
    np.testing.assert_allclose(float(val), float(jnp.cos(1.234)), atol=1e-5)
    np.testing.assert_allclose(float(grad["a"]), float(-jnp.sin(1.234)), atol=1e-5)


@pytest.mark.parametrize("theta", [0.1, 0.7, -0.5, 1.5])
def test_parameter_shift_matches_grad(theta: float) -> None:
    circuit = Circuit(1).ry(0, "a")
    hamiltonian = PauliHamiltonian([("Z", 1.0)])
    params = {"a": jnp.array(theta)}

    def loss(p):
        return _energy(p, circuit, hamiltonian)

    ps = parameter_shift_grad(loss, params, "a")
    auto = jax.grad(loss)(params)["a"]
    np.testing.assert_allclose(float(ps), float(auto), atol=1e-4)


def test_jit_grad_matches_eager_grad() -> None:
    circuit = Circuit(2).ry(0, "a").cx(0, 1).rz(1, "b")
    hamiltonian = PauliHamiltonian([("ZZ", 1.0)])
    params = {"a": jnp.array(0.4), "b": jnp.array(-1.1)}

    eager_grad = jax.grad(_energy)(params, circuit, hamiltonian)
    jit_grad = jax.jit(jax.grad(_energy), static_argnums=(1, 2))(params, circuit, hamiltonian)
    for k in eager_grad:
        np.testing.assert_allclose(
            np.asarray(eager_grad[k]), np.asarray(jit_grad[k]), atol=1e-5
        )
