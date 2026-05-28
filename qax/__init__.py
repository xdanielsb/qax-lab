"""QAX Lab: differentiable quantum circuits in JAX."""

from __future__ import annotations

from . import gates, metrics, noise, optim, qasm, sampling
from .circuit import Circuit, GateOp
from .metrics import fidelity, infidelity, inner_product
from .noise import NoiseChannel, apply_channel
from .observables import PauliHamiltonian, PauliTerm, apply_pauli_string, expectation
from .program import Program, ProgramOp, compile_circuit
from .qasm import to_qasm
from .sampling import counts_from_indices, sample_counts, sample_indices
from .simulate import apply_unitaries, make_unitaries, simulate, simulate_scan
from .state import basis_state, bitstring, normalize, probabilities, zero_state

__all__ = [
    "Circuit",
    "GateOp",
    "PauliHamiltonian",
    "PauliTerm",
    "Program",
    "ProgramOp",
    "NoiseChannel",
    "apply_channel",
    "apply_pauli_string",
    "apply_unitaries",
    "basis_state",
    "bitstring",
    "compile_circuit",
    "counts_from_indices",
    "expectation",
    "fidelity",
    "gates",
    "infidelity",
    "inner_product",
    "make_unitaries",
    "metrics",
    "noise",
    "normalize",
    "optim",
    "probabilities",
    "qasm",
    "sample_counts",
    "sample_indices",
    "sampling",
    "simulate",
    "simulate_scan",
    "to_qasm",
    "zero_state",
]

__version__ = "0.1.0"
