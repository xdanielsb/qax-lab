"""Pauli observables and Hamiltonian expectation values.

A ``PauliHamiltonian`` is a (real-)weighted sum of Pauli strings,

    H = sum_j c_j * P_j

where each ``P_j`` is a tensor product over qubits drawn from
``{I, X, Y, Z}``. We compute ``<psi|H|psi>`` by applying each ``P_j`` to the
statevector using the same tensor-contraction kernel as :func:`simulate`,
then taking the inner product with the original state. This keeps the cost
linear in the number of terms and avoids materializing the full Hamiltonian
matrix.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from . import gates
from .simulate import _apply_gate
from .typing import DEFAULT_COMPLEX, Array

_PAULI_MATRICES: dict[str, callable] = {
    "I": gates.i,
    "X": gates.x,
    "Y": gates.y,
    "Z": gates.z,
}


@dataclass(frozen=True)
class PauliTerm:
    """A single weighted Pauli tensor product like ``("XYZ", -0.25)``."""

    pauli: str
    coeff: float

    def __post_init__(self) -> None:
        if not self.pauli or any(ch not in _PAULI_MATRICES for ch in self.pauli):
            raise ValueError(
                f"Invalid Pauli string {self.pauli!r}; must be non-empty and only contain I/X/Y/Z."
            )


@dataclass(frozen=True)
class PauliHamiltonian:
    """A real-weighted sum of Pauli-string terms.

    Examples
    --------
    >>> from qax import PauliHamiltonian
    >>> PauliHamiltonian([("ZI", 1.0), ("IZ", 1.0), ("XX", -0.5)])
    """

    terms: tuple[PauliTerm, ...]

    def __init__(self, terms: Iterable[tuple[str, float] | PauliTerm]):
        normalized: list[PauliTerm] = []
        width: int | None = None
        for t in terms:
            if isinstance(t, PauliTerm):
                term = t
            else:
                pauli, coeff = t
                term = PauliTerm(pauli=pauli, coeff=float(coeff))
            if width is None:
                width = len(term.pauli)
            elif len(term.pauli) != width:
                raise ValueError(
                    f"All Pauli strings must have the same width; got {len(term.pauli)} vs {width}."
                )
            normalized.append(term)
        if width is None:
            raise ValueError("PauliHamiltonian needs at least one term.")
        object.__setattr__(self, "terms", tuple(normalized))

    @property
    def n_qubits(self) -> int:
        return len(self.terms[0].pauli)

    def __iter__(self):
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)


def apply_pauli_string(state: Array, pauli: str, dtype=DEFAULT_COMPLEX) -> Array:
    """Apply a tensor-product Pauli string to ``state``."""
    n = len(pauli)
    if state.shape != (1 << n,):
        raise ValueError(
            f"Pauli string of width {n} expects state of dim {1 << n}, got {state.shape}."
        )
    out = state
    for wire, ch in enumerate(pauli):
        if ch == "I":
            continue
        gate = _PAULI_MATRICES[ch](dtype=dtype)
        out = _apply_gate(out, gate, (wire,), n)
    return out


def expectation(state: Array, hamiltonian: PauliHamiltonian, dtype=DEFAULT_COMPLEX) -> Array:
    """Compute ``<psi|H|psi>`` as a real-valued JAX scalar.

    Implementation note: we compute each term as a separate
    ``<psi|P|psi> * coeff`` contribution. JAX will fuse them under JIT.
    """
    if state.shape != (1 << hamiltonian.n_qubits,):
        raise ValueError(
            f"Hamiltonian on {hamiltonian.n_qubits} qubits cannot act on state "
            f"of shape {state.shape}."
        )

    total = jnp.asarray(0.0, dtype=jnp.float32 if dtype == jnp.complex64 else jnp.float64)
    for term in hamiltonian.terms:
        applied = apply_pauli_string(state, term.pauli, dtype=dtype)
        # <psi|P|psi> is real for any Hermitian Pauli string, so we can
        # safely take .real before summing.
        amp = jnp.vdot(state, applied).real
        total = total + term.coeff * amp
    return total


@jax.jit
def expectation_pauli(state: Array, pauli_index: Array, coeffs: Array) -> Array:
    """Vectorized expectation for a stack of Pauli strings encoded as int arrays.

    Each row of ``pauli_index`` carries integer codes for the per-qubit Pauli
    (0=I, 1=X, 2=Y, 3=Z); ``coeffs`` carries the matching real weights. This
    path is here for users who want to push a whole Hamiltonian through
    ``vmap``/``scan`` once it has been packed.
    """
    # Materialize per-qubit Pauli matrices once and gather by index.
    pauli_stack = jnp.stack([gates.i(), gates.x(), gates.y(), gates.z()], axis=0)  # shape (4, 2, 2)

    n_qubits = pauli_index.shape[1]

    def term_value(row: Array, coeff: Array) -> Array:
        out = state
        for wire in range(n_qubits):
            gate = pauli_stack[row[wire]]
            out = _apply_gate(out, gate, (wire,), n_qubits)
        return coeff * jnp.vdot(state, out).real

    values = jax.vmap(term_value)(pauli_index, coeffs)
    return values.sum()
