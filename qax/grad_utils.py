"""Educational comparisons against :func:`jax.grad`.

Right now we only expose the parameter-shift rule, which lets you compute
the gradient of an expectation value w.r.t. a single ``RX``/``RY``/``RZ``
angle by evaluating the expectation at ``theta +/- pi/2`` and taking half
the difference. For the supported gate set this gives results identical to
``jax.grad`` up to floating-point error.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from .typing import Array, ParamTree


def parameter_shift_grad(
    loss_fn: Callable[[ParamTree], Array],
    params: ParamTree,
    name: str,
    shift: float = jnp.pi / 2,
) -> Array:
    """Compute d(loss_fn)/d(params[name]) via the parameter-shift rule.

    Notes
    -----
    Only valid for gates whose generator has eigenvalues ``+/- 1/2``
    (the Pauli rotations ``RX``, ``RY``, ``RZ``). The standard prefactor is
    ``1/2 * (f(theta + pi/2) - f(theta - pi/2))``.
    """
    if name not in params:
        raise KeyError(f"Parameter {name!r} not in params.")

    plus = dict(params)
    plus[name] = params[name] + shift
    minus = dict(params)
    minus[name] = params[name] - shift

    return 0.5 * (loss_fn(plus) - loss_fn(minus))
