"""State-comparison metrics: fidelity, trace distance."""

from __future__ import annotations

import jax.numpy as jnp

from .typing import Array


def inner_product(a: Array, b: Array) -> Array:
    """Complex inner product ``<a|b>``."""
    return jnp.vdot(a, b)


def fidelity(a: Array, b: Array) -> Array:
    """Pure-state fidelity ``|<a|b>|^2``. Real-valued JAX scalar in ``[0, 1]``."""
    return jnp.abs(jnp.vdot(a, b)) ** 2


def infidelity(a: Array, b: Array) -> Array:
    return 1.0 - fidelity(a, b)


def trace_distance_pure(a: Array, b: Array) -> Array:
    """Trace distance for pure states: ``sqrt(1 - |<a|b>|^2)``."""
    return jnp.sqrt(jnp.clip(1.0 - fidelity(a, b), a_min=0.0, a_max=1.0))


def l2_distance(a: Array, b: Array) -> Array:
    """Plain Euclidean distance — handy as an unbiased differentiable loss."""
    diff = a - b
    return jnp.sqrt(jnp.sum((diff.conj() * diff).real))
