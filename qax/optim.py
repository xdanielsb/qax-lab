"""Hand-rolled optimizers (SGD and Adam).

The point of writing these from scratch (rather than depending on Optax) is
to make the JAX update logic visible — every transformation is a pure
function on PyTrees of arrays.

Both optimizers expose the same minimal interface::

    opt = adam(learning_rate=1e-2)
    state = opt.init(params)
    # ...
    updates, state = opt.update(grads, state, params)
    params = opt.apply_updates(params, updates)

``init``, ``update``, and ``apply_updates`` are all pure functions that
return new pytrees, which keeps everything JIT-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .typing import Array, ParamTree

PyTreeOf = ParamTree


@dataclass(frozen=True)
class Optimizer:
    """Tiny PyTree-friendly optimizer protocol.

    ``init`` builds a state pytree from the parameters; ``update`` consumes
    gradients and returns ``(updates, new_state)``; ``apply_updates`` adds
    the updates to the params. The methods are all pure functions of their
    arguments.
    """

    init: Callable[[ParamTree], Any]
    update: Callable[[ParamTree, Any, ParamTree], tuple[ParamTree, Any]]
    apply_updates: Callable[[ParamTree, ParamTree], ParamTree]


def _tree_axpy(alpha: float, x: ParamTree, y: ParamTree) -> ParamTree:
    return jax.tree.map(lambda xi, yi: alpha * xi + yi, x, y)


def _zeros_like(tree: ParamTree) -> ParamTree:
    return jax.tree.map(jnp.zeros_like, tree)


# ---------------------------------------------------------------------------
# SGD (with optional momentum)
# ---------------------------------------------------------------------------

def sgd(learning_rate: float, momentum: float = 0.0) -> Optimizer:
    """Stochastic gradient descent with optional Polyak momentum."""

    def init(params: ParamTree) -> ParamTree:
        return _zeros_like(params)  # velocity buffer

    def update(grads: ParamTree, state: ParamTree, params: ParamTree):
        del params
        if momentum == 0.0:
            updates = jax.tree.map(lambda g: -learning_rate * g, grads)
            return updates, state
        new_velocity = jax.tree.map(lambda v, g: momentum * v + g, state, grads)
        updates = jax.tree.map(lambda v: -learning_rate * v, new_velocity)
        return updates, new_velocity

    def apply_updates(params: ParamTree, updates: ParamTree) -> ParamTree:
        return jax.tree.map(lambda p, u: p + u, params, updates)

    return Optimizer(init=init, update=update, apply_updates=apply_updates)


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdamState:
    """First/second-moment buffers and step counter."""
    step: Array
    m: ParamTree
    v: ParamTree


# Register so JAX can ``jit`` through it and ``tree.map`` over it.
jax.tree_util.register_pytree_node(
    AdamState,
    lambda s: ((s.step, s.m, s.v), None),
    lambda _, children: AdamState(step=children[0], m=children[1], v=children[2]),
)


def adam(
    learning_rate: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> Optimizer:
    """Standard Adam (Kingma & Ba, 2014) with bias correction."""

    def init(params: ParamTree) -> AdamState:
        return AdamState(
            step=jnp.zeros((), dtype=jnp.int32),
            m=_zeros_like(params),
            v=_zeros_like(params),
        )

    def update(grads: ParamTree, state: AdamState, params: ParamTree):
        del params
        step = state.step + 1
        new_m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state.m, grads)
        new_v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * (g * g), state.v, grads)
        # Bias correction.
        step_f = step.astype(jnp.float32)
        bc1 = 1.0 - jnp.power(b1, step_f)
        bc2 = 1.0 - jnp.power(b2, step_f)
        updates = jax.tree.map(
            lambda m, v: -learning_rate * (m / bc1) / (jnp.sqrt(v / bc2) + eps),
            new_m,
            new_v,
        )
        return updates, AdamState(step=step, m=new_m, v=new_v)

    def apply_updates(params: ParamTree, updates: ParamTree) -> ParamTree:
        return jax.tree.map(lambda p, u: p + u, params, updates)

    return Optimizer(init=init, update=update, apply_updates=apply_updates)
