"""Tests for the hand-rolled optimizers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qax.optim import adam, sgd


def _quadratic(params):
    # Simple convex loss: sum_i (p_i - target_i)^2 with target = [1, -2, 0.5].
    target = jnp.array([1.0, -2.0, 0.5])
    return jnp.sum((params["x"] - target) ** 2)


@pytest.mark.parametrize("optimizer_factory", [lambda: sgd(0.1), lambda: adam(0.1)])
def test_optimizer_decreases_loss(optimizer_factory) -> None:
    opt = optimizer_factory()
    params = {"x": jnp.array([0.0, 0.0, 0.0])}
    state = opt.init(params)

    losses = []
    for _ in range(200):
        loss, grads = jax.value_and_grad(_quadratic)(params)
        updates, state = opt.update(grads, state, params)
        params = opt.apply_updates(params, updates)
        losses.append(float(loss))

    assert losses[-1] < losses[0]
    assert losses[-1] < 1e-3


def test_adam_reaches_optimum_close() -> None:
    opt = adam(0.1)
    params = {"x": jnp.array([0.0, 0.0, 0.0])}
    state = opt.init(params)
    for _ in range(500):
        _, grads = jax.value_and_grad(_quadratic)(params)
        updates, state = opt.update(grads, state, params)
        params = opt.apply_updates(params, updates)
    np.testing.assert_allclose(np.asarray(params["x"]), [1.0, -2.0, 0.5], atol=1e-3)


def test_sgd_with_momentum_decreases_loss() -> None:
    opt = sgd(0.05, momentum=0.9)
    params = {"x": jnp.array([0.0, 0.0, 0.0])}
    state = opt.init(params)
    initial = float(_quadratic(params))
    for _ in range(200):
        _, grads = jax.value_and_grad(_quadratic)(params)
        updates, state = opt.update(grads, state, params)
        params = opt.apply_updates(params, updates)
    assert float(_quadratic(params)) < initial


def test_train_step_is_jittable() -> None:
    opt = adam(0.05)
    params = {"x": jnp.array([0.0, 0.0, 0.0])}
    state = opt.init(params)

    @jax.jit
    def step(params, state):
        loss, grads = jax.value_and_grad(_quadratic)(params)
        updates, state = opt.update(grads, state, params)
        params = opt.apply_updates(params, updates)
        return params, state, loss

    for _ in range(50):
        params, state, _ = step(params, state)
    assert float(_quadratic(params)) < 1.0
