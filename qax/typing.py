"""Type aliases used across QAX Lab.

We keep these in one place so signatures everywhere stay readable.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

Array = jax.Array

# A scalar parameter can be a Python float or a JAX/NumPy scalar.
Scalar = float | int | Array

# A pytree of parameters. Typically a dict[str, Array].
ParamTree = Any

# An explicit JAX PRNG key.
PRNGKey = Array

# Default dtypes. Statevectors are complex; parameters are real.
DEFAULT_COMPLEX = jnp.complex64
DEFAULT_REAL = jnp.float32
