"""Compiled, JIT-friendly representation of a ``Circuit``.

A ``Program`` is a frozen, hashable bundle of:

  * static metadata (qubit count, gate names, wires, which parameter feeds each
    gate) — fine to pass as a static argument to :func:`jax.jit`;
  * a list of parameter names referenced by the circuit.

We deliberately keep the parameter *values* out of the ``Program`` so the
caller can pass them as a separate PyTree, which is what JAX needs to
differentiate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp

from .circuit import Circuit
from .gates import PARAMETERIZED
from .typing import Array, ParamTree


@dataclass(frozen=True)
class ProgramOp:
    """A frozen, hashable description of one compiled gate."""
    name: str
    wires: tuple[int, ...]
    param_name: str | None     # name in the user's params dict, if any
    param_value: float | None  # baked-in literal, if any


@dataclass(frozen=True)
class Program:
    """A compiled, hashable circuit representation."""

    n_qubits: int
    ops: tuple[ProgramOp, ...]
    parameter_names: tuple[str, ...]

    # The ``Program`` is intentionally hash/eq-friendly so that JAX can use it
    # as a static argument and avoid recompiling for equal circuits.

    @property
    def dim(self) -> int:
        return 1 << self.n_qubits

    @property
    def depth(self) -> int:
        return len(self.ops)


def compile_circuit(circuit: Circuit) -> Program:
    """Validate ``circuit`` and freeze it into a ``Program``.

    Any string parameter that appears on a parameterized gate becomes a
    *symbolic* slot; numeric parameter values get baked in as literals so they
    flow through the static-argument fast path.
    """
    if circuit.n_qubits < 1:
        raise ValueError("Circuit must have at least one qubit.")

    ops: list[ProgramOp] = []
    seen_params: dict[str, None] = {}

    for op in circuit.ops:
        if op.name in PARAMETERIZED:
            if isinstance(op.param, str):
                seen_params[op.param] = None
                ops.append(
                    ProgramOp(
                        name=op.name,
                        wires=op.wires,
                        param_name=op.param,
                        param_value=None,
                    )
                )
            else:
                ops.append(
                    ProgramOp(
                        name=op.name,
                        wires=op.wires,
                        param_name=None,
                        param_value=float(op.param) if op.param is not None else None,
                    )
                )
        else:
            ops.append(
                ProgramOp(
                    name=op.name,
                    wires=op.wires,
                    param_name=None,
                    param_value=None,
                )
            )

    return Program(
        n_qubits=circuit.n_qubits,
        ops=tuple(ops),
        parameter_names=tuple(seen_params.keys()),
    )


def resolve_parameter(op: ProgramOp, params: ParamTree | None) -> Array | None:
    """Return the JAX scalar value for ``op``'s parameter, if any."""
    if op.param_name is not None:
        if not isinstance(params, Mapping) or op.param_name not in params:
            raise KeyError(
                f"Missing parameter {op.param_name!r}; circuit references parameters "
                f"but they were not supplied."
            )
        return jnp.asarray(params[op.param_name])
    if op.param_value is not None:
        return jnp.asarray(op.param_value)
    return None


def assert_params_match(program: Program, params: ParamTree | None) -> None:
    """Raise if ``params`` is missing names the program needs."""
    if not program.parameter_names:
        return
    if params is None:
        raise KeyError(
            f"Circuit needs parameters {list(program.parameter_names)} but none were given."
        )
    if not isinstance(params, Mapping):
        raise TypeError(
            f"params must be a mapping with keys {list(program.parameter_names)}, "
            f"got {type(params).__name__}."
        )
    missing = [n for n in program.parameter_names if n not in params]
    if missing:
        raise KeyError(f"Missing parameters: {missing}.")
