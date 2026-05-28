"""Dense statevector simulation.

The implementation prefers a *tensor-contraction* style: each gate is applied
by reshaping the statevector into a ``(2,) * n`` tensor, contracting the gate
axes against the appropriate qubit dimensions, and reshaping back. This keeps
memory at ``O(2^n)`` rather than building full ``2^n x 2^n`` operators, which
matters at 8+ qubits.

The same module also exposes a ``lax.scan`` backend that operates on
*pre-stacked full-system unitaries* for cases where you want the simplest
possible compiled-loop story (e.g. for benchmarking).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .circuit import Circuit
from .gates import gate_matrix
from .program import Program, ProgramOp, assert_params_match, compile_circuit, resolve_parameter
from .state import zero_state
from .typing import Array, DEFAULT_COMPLEX, ParamTree


# ---------------------------------------------------------------------------
# Tensor-contraction kernel
# ---------------------------------------------------------------------------

def _apply_gate(state: Array, gate: Array, wires: tuple[int, ...], n_qubits: int) -> Array:
    """Apply ``gate`` (shape ``(2^k, 2^k)``) to ``wires`` of ``state``.

    Implementation uses ``jnp.tensordot`` + ``moveaxis``; no full-system
    matrix is ever materialized.
    """
    k = len(wires)
    if gate.shape != (1 << k, 1 << k):
        raise ValueError(
            f"Gate shape {gate.shape} does not match {k} wires."
        )

    # Reshape statevector to a 2 x 2 x ... x 2 tensor and the gate to
    # (2,)*k_in x (2,)*k_out. The contraction is over the "input" axes of the
    # gate against the qubit axes in ``wires``.
    state_t = state.reshape((2,) * n_qubits)
    gate_t = gate.reshape((2,) * k + (2,) * k)

    # tensordot contracts the *last* k axes of ``gate_t`` with the listed axes
    # of ``state_t``.
    contracted = jnp.tensordot(gate_t, state_t, axes=(list(range(k, 2 * k)), list(wires)))
    # After tensordot, the first k axes correspond to the *new* values on
    # ``wires``; everything else keeps its original order with the wire axes
    # removed. Move them back into their original positions.
    remaining = [ax for ax in range(n_qubits) if ax not in wires]
    src = list(range(k))                     # output qubit axes (currently in front)
    dst = list(wires)                        # where they should end up
    # Restore the canonical [0, 1, ..., n-1] order.
    out = jnp.moveaxis(contracted, src, dst)
    # ``out`` now has axes laid out as [original axis 0, 1, ..., n-1] with the
    # wire axes replaced by their new values, while ``remaining`` axes carry
    # their original data. ``moveaxis`` does the right thing because tensordot
    # placed the new wire axes at the front in the order given by ``wires``.
    del remaining
    return out.reshape((1 << n_qubits,))


def _apply_program_op(state: Array, op: ProgramOp, params: ParamTree | None, n_qubits: int,
                      dtype) -> Array:
    param_value = resolve_parameter(op, params)
    if param_value is None:
        gate = gate_matrix(op.name, dtype=dtype)
    else:
        gate = gate_matrix(op.name, param_value, dtype=dtype)
    return _apply_gate(state, gate, op.wires, n_qubits)


# ---------------------------------------------------------------------------
# Top-level eager / JIT simulation
# ---------------------------------------------------------------------------

def _simulate_impl(program: Program, params: ParamTree | None, initial_state: Array | None,
                   dtype) -> Array:
    state = zero_state(program.n_qubits, dtype=dtype) if initial_state is None else initial_state
    for op in program.ops:
        state = _apply_program_op(state, op, params, program.n_qubits, dtype)
    return state


@partial(jax.jit, static_argnames=("program", "dtype"))
def _simulate_jit(program: Program, params: ParamTree | None, initial_state: Array | None,
                  dtype) -> Array:
    return _simulate_impl(program, params, initial_state, dtype)


def simulate(
    circuit_or_program: Circuit | Program,
    params: ParamTree | None = None,
    initial_state: Array | None = None,
    *,
    jit: bool = True,
    dtype=DEFAULT_COMPLEX,
) -> Array:
    """Run a circuit and return the final statevector.

    Parameters
    ----------
    circuit_or_program:
        A user-facing ``Circuit`` or a pre-compiled ``Program``. Compiling
        once and reusing across many ``simulate`` calls lets JAX cache its
        compiled traces.
    params:
        A pytree (typically a ``dict[str, jnp.ndarray]``) holding the values
        for any symbolic parameters in the circuit. Required only if the
        circuit references symbolic parameters.
    initial_state:
        Optional starting statevector. Defaults to ``|0...0>``.
    jit:
        Wrap the body in :func:`jax.jit` (default). Set to ``False`` to keep
        the loop eager — useful for benchmarking or debugging.
    dtype:
        Complex dtype for the statevector. Defaults to ``complex64``.
    """
    program = (
        compile_circuit(circuit_or_program) if isinstance(circuit_or_program, Circuit)
        else circuit_or_program
    )
    assert_params_match(program, params)
    if jit:
        return _simulate_jit(program, params, initial_state, dtype)
    return _simulate_impl(program, params, initial_state, dtype)


# ---------------------------------------------------------------------------
# lax.scan backend (full-system unitaries)
# ---------------------------------------------------------------------------

def _embed_full_unitary(gate: Array, wires: tuple[int, ...], n_qubits: int, dtype) -> Array:
    """Lift a small gate up to the full ``2^n x 2^n`` operator.

    This is intentionally the slow/simple path: it is here so we can stack
    per-gate unitaries and feed them to ``lax.scan`` as a single ``Array`` of
    shape ``(depth, dim, dim)``. For real workloads, prefer
    :func:`simulate`, which contracts gates directly against the statevector.
    """
    dim = 1 << n_qubits
    eye_basis = jnp.eye(dim, dtype=dtype)
    # Apply the gate to each column of the identity using the tensor-contraction
    # kernel; that gives us the gate's matrix in the full basis.
    cols = jax.vmap(lambda col: _apply_gate(col, gate, wires, n_qubits), in_axes=1, out_axes=1)
    return cols(eye_basis)


def make_unitaries(program: Program, params: ParamTree | None, dtype=DEFAULT_COMPLEX) -> Array:
    """Build a ``(depth, dim, dim)`` stack of full-system unitaries.

    Only valid when every op has a *fixed* shape contribution, which is the
    case for the supported gate set.
    """
    assert_params_match(program, params)
    mats: list[Array] = []
    for op in program.ops:
        param_value = resolve_parameter(op, params)
        if param_value is None:
            local = gate_matrix(op.name, dtype=dtype)
        else:
            local = gate_matrix(op.name, param_value, dtype=dtype)
        mats.append(_embed_full_unitary(local, op.wires, program.n_qubits, dtype))
    if not mats:
        return jnp.zeros((0, program.dim, program.dim), dtype=dtype)
    return jnp.stack(mats, axis=0)


@jax.jit
def apply_unitaries(unitaries: Array, initial_state: Array) -> Array:
    """Apply each unitary in turn to ``initial_state`` using ``jax.lax.scan``."""
    def step(state: Array, u: Array) -> tuple[Array, None]:
        return u @ state, None

    final_state, _ = jax.lax.scan(step, initial_state, unitaries)
    return final_state


def simulate_scan(
    circuit_or_program: Circuit | Program,
    params: ParamTree | None = None,
    initial_state: Array | None = None,
    *,
    dtype=DEFAULT_COMPLEX,
) -> Array:
    """Scan-based simulation. Slower than :func:`simulate` but pedagogically clear."""
    program = (
        compile_circuit(circuit_or_program) if isinstance(circuit_or_program, Circuit)
        else circuit_or_program
    )
    state = zero_state(program.n_qubits, dtype=dtype) if initial_state is None else initial_state
    unitaries = make_unitaries(program, params, dtype=dtype)
    if unitaries.shape[0] == 0:
        return state
    return apply_unitaries(unitaries, state)
