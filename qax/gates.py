"""Standard gate matrices.

Each function returns a JAX array. Parameterized gates take a real-valued
``theta`` (a Python float, NumPy scalar, or 0-d JAX array) and return a
complex unitary. Gate matrices are unitary up to numerical tolerance.

The matrix conventions follow the usual textbook ordering: the leftmost
qubit in a multi-qubit gate is the most significant bit of the basis-state
index, i.e. ``|q0 q1>`` corresponds to index ``2*q0 + q1``.
"""

from __future__ import annotations

import jax.numpy as jnp

from .typing import DEFAULT_COMPLEX, Array, Scalar

# ---------------------------------------------------------------------------
# Constant single-qubit gates
# ---------------------------------------------------------------------------


def i(dtype=DEFAULT_COMPLEX) -> Array:
    return jnp.eye(2, dtype=dtype)


def x(dtype=DEFAULT_COMPLEX) -> Array:
    return jnp.array([[0, 1], [1, 0]], dtype=dtype)


def y(dtype=DEFAULT_COMPLEX) -> Array:
    return jnp.array([[0, -1j], [1j, 0]], dtype=dtype)


def z(dtype=DEFAULT_COMPLEX) -> Array:
    return jnp.array([[1, 0], [0, -1]], dtype=dtype)


def h(dtype=DEFAULT_COMPLEX) -> Array:
    return jnp.array([[1, 1], [1, -1]], dtype=dtype) / jnp.sqrt(jnp.asarray(2, dtype=dtype))


def s(dtype=DEFAULT_COMPLEX) -> Array:
    return jnp.array([[1, 0], [0, 1j]], dtype=dtype)


def sdg(dtype=DEFAULT_COMPLEX) -> Array:
    return jnp.array([[1, 0], [0, -1j]], dtype=dtype)


def t(dtype=DEFAULT_COMPLEX) -> Array:
    return jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]], dtype=dtype)


def tdg(dtype=DEFAULT_COMPLEX) -> Array:
    return jnp.array([[1, 0], [0, jnp.exp(-1j * jnp.pi / 4)]], dtype=dtype)


# ---------------------------------------------------------------------------
# Parameterized single-qubit gates
# ---------------------------------------------------------------------------


def _theta(theta: Scalar, dtype) -> Array:
    # Promote to the gate's real component dtype before any complex ops so that
    # gradients flow with the expected precision.
    real_dtype = jnp.float64 if dtype == jnp.complex128 else jnp.float32
    return jnp.asarray(theta, dtype=real_dtype)


def _stack_2x2(a00: Array, a01: Array, a10: Array, a11: Array, dtype) -> Array:
    """Pack four scalar (possibly traced) values into a 2x2 matrix.

    We go through ``jnp.stack`` so the function plays nicely with tracers
    inside ``jit``/``vmap``/``grad``.
    """
    row0 = jnp.stack([a00.astype(dtype), a01.astype(dtype)])
    row1 = jnp.stack([a10.astype(dtype), a11.astype(dtype)])
    return jnp.stack([row0, row1])


def rx(theta: Scalar, dtype=DEFAULT_COMPLEX) -> Array:
    th = _theta(theta, dtype)
    c = jnp.cos(th / 2)
    s_ = jnp.sin(th / 2)
    zero = jnp.zeros_like(c)
    return _stack_2x2(c, -1j * s_, -1j * s_, c + zero, dtype)


def ry(theta: Scalar, dtype=DEFAULT_COMPLEX) -> Array:
    th = _theta(theta, dtype)
    c = jnp.cos(th / 2)
    s_ = jnp.sin(th / 2)
    return _stack_2x2(c, -s_, s_, c, dtype)


def rz(theta: Scalar, dtype=DEFAULT_COMPLEX) -> Array:
    th = _theta(theta, dtype)
    e_minus = jnp.exp(-1j * th / 2)
    e_plus = jnp.exp(1j * th / 2)
    zero = jnp.zeros_like(e_minus)
    return _stack_2x2(e_minus, zero, zero, e_plus, dtype)


def phase(theta: Scalar, dtype=DEFAULT_COMPLEX) -> Array:
    """Phase gate diag(1, e^{i*theta})."""
    th = _theta(theta, dtype)
    e_plus = jnp.exp(1j * th)
    one = jnp.ones_like(e_plus)
    zero = jnp.zeros_like(e_plus)
    return _stack_2x2(one, zero, zero, e_plus, dtype)


# ---------------------------------------------------------------------------
# Two-qubit gates
# ---------------------------------------------------------------------------


def cnot(dtype=DEFAULT_COMPLEX) -> Array:
    """CNOT with qubit 0 as control, qubit 1 as target (big-endian)."""
    return jnp.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=dtype,
    )


def cz(dtype=DEFAULT_COMPLEX) -> Array:
    return jnp.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ],
        dtype=dtype,
    )


def swap(dtype=DEFAULT_COMPLEX) -> Array:
    return jnp.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=dtype,
    )


def crx(theta: Scalar, dtype=DEFAULT_COMPLEX) -> Array:
    return _controlled(rx(theta, dtype=dtype), dtype=dtype)


def cry(theta: Scalar, dtype=DEFAULT_COMPLEX) -> Array:
    return _controlled(ry(theta, dtype=dtype), dtype=dtype)


def crz(theta: Scalar, dtype=DEFAULT_COMPLEX) -> Array:
    return _controlled(rz(theta, dtype=dtype), dtype=dtype)


def _controlled(u: Array, dtype) -> Array:
    """Build the 4x4 controlled-U with the first qubit as control."""
    top = jnp.eye(2, dtype=dtype)
    out = jnp.zeros((4, 4), dtype=dtype)
    out = out.at[:2, :2].set(top)
    out = out.at[2:, 2:].set(u.astype(dtype))
    return out


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

CONSTANT_GATES: dict[str, callable] = {
    "i": i,
    "x": x,
    "y": y,
    "z": z,
    "h": h,
    "s": s,
    "sdg": sdg,
    "t": t,
    "tdg": tdg,
    "cnot": cnot,
    "cx": cnot,  # alias
    "cz": cz,
    "swap": swap,
}

PARAM_GATES: dict[str, callable] = {
    "rx": rx,
    "ry": ry,
    "rz": rz,
    "phase": phase,
    "crx": crx,
    "cry": cry,
    "crz": crz,
}

GATE_NUM_WIRES: dict[str, int] = {
    "i": 1,
    "x": 1,
    "y": 1,
    "z": 1,
    "h": 1,
    "s": 1,
    "sdg": 1,
    "t": 1,
    "tdg": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "phase": 1,
    "cnot": 2,
    "cx": 2,
    "cz": 2,
    "swap": 2,
    "crx": 2,
    "cry": 2,
    "crz": 2,
}

PARAMETERIZED: set[str] = set(PARAM_GATES.keys())


def gate_matrix(name: str, param_value: Scalar | None = None, dtype=DEFAULT_COMPLEX) -> Array:
    """Return the matrix for ``name``. ``param_value`` is required iff parameterized."""
    name = name.lower()
    if name in CONSTANT_GATES:
        return CONSTANT_GATES[name](dtype=dtype)
    if name in PARAM_GATES:
        if param_value is None:
            raise ValueError(f"Gate {name!r} is parameterized; a parameter value is required.")
        return PARAM_GATES[name](param_value, dtype=dtype)
    raise ValueError(f"Unknown gate {name!r}.")


def num_wires(name: str) -> int:
    name = name.lower()
    if name not in GATE_NUM_WIRES:
        raise ValueError(f"Unknown gate {name!r}.")
    return GATE_NUM_WIRES[name]


def is_parameterized(name: str) -> bool:
    return name.lower() in PARAMETERIZED
