"""Immutable user-facing ``Circuit`` builder.

A ``Circuit`` is just a tuple of ``GateOp``s plus the qubit count. Methods
like ``.h(0)`` and ``.cx(0, 1)`` return a *new* ``Circuit`` rather than
mutating in place, which makes the type trivially hashable and safe to pass
as a static argument to :func:`jax.jit`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .gates import GATE_NUM_WIRES, PARAMETERIZED


@dataclass(frozen=True)
class GateOp:
    """A single gate application."""
    name: str
    wires: tuple[int, ...]
    param: str | float | None = None

    def __post_init__(self) -> None:
        name = self.name.lower()
        if name not in GATE_NUM_WIRES:
            raise ValueError(f"Unknown gate {self.name!r}.")
        expected = GATE_NUM_WIRES[name]
        if len(self.wires) != expected:
            raise ValueError(
                f"Gate {name!r} acts on {expected} wires, got {len(self.wires)}: {self.wires!r}."
            )
        if len(set(self.wires)) != len(self.wires):
            raise ValueError(f"Gate {name!r} cannot repeat wires: {self.wires!r}.")
        if name in PARAMETERIZED and self.param is None:
            raise ValueError(f"Gate {name!r} requires a parameter.")
        if name not in PARAMETERIZED and self.param is not None:
            raise ValueError(f"Gate {name!r} does not take a parameter.")
        # Normalize stored name to lowercase.
        object.__setattr__(self, "name", name)


@dataclass(frozen=True)
class Circuit:
    """An immutable sequence of gate operations on a fixed qubit register."""

    n_qubits: int
    ops: tuple[GateOp, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _add(self, op: GateOp) -> "Circuit":
        for w in op.wires:
            if not 0 <= w < self.n_qubits:
                raise ValueError(
                    f"Wire {w} out of range for circuit with {self.n_qubits} qubits."
                )
        return replace(self, ops=self.ops + (op,))

    def add(self, name: str, wires: tuple[int, ...], param: str | float | None = None) -> "Circuit":
        """General-purpose append; mainly useful for programmatic circuit building."""
        return self._add(GateOp(name=name, wires=tuple(wires), param=param))

    # ------------------------------------------------------------------
    # constant single-qubit gates
    # ------------------------------------------------------------------

    def i(self, w: int) -> "Circuit":
        return self._add(GateOp("i", (w,)))

    def x(self, w: int) -> "Circuit":
        return self._add(GateOp("x", (w,)))

    def y(self, w: int) -> "Circuit":
        return self._add(GateOp("y", (w,)))

    def z(self, w: int) -> "Circuit":
        return self._add(GateOp("z", (w,)))

    def h(self, w: int) -> "Circuit":
        return self._add(GateOp("h", (w,)))

    def s(self, w: int) -> "Circuit":
        return self._add(GateOp("s", (w,)))

    def sdg(self, w: int) -> "Circuit":
        return self._add(GateOp("sdg", (w,)))

    def t(self, w: int) -> "Circuit":
        return self._add(GateOp("t", (w,)))

    def tdg(self, w: int) -> "Circuit":
        return self._add(GateOp("tdg", (w,)))

    # ------------------------------------------------------------------
    # parameterized single-qubit gates
    # ------------------------------------------------------------------

    def rx(self, w: int, param: str | float) -> "Circuit":
        return self._add(GateOp("rx", (w,), param))

    def ry(self, w: int, param: str | float) -> "Circuit":
        return self._add(GateOp("ry", (w,), param))

    def rz(self, w: int, param: str | float) -> "Circuit":
        return self._add(GateOp("rz", (w,), param))

    def phase(self, w: int, param: str | float) -> "Circuit":
        return self._add(GateOp("phase", (w,), param))

    # ------------------------------------------------------------------
    # two-qubit gates
    # ------------------------------------------------------------------

    def cx(self, control: int, target: int) -> "Circuit":
        return self._add(GateOp("cx", (control, target)))

    cnot = cx

    def cz(self, q0: int, q1: int) -> "Circuit":
        return self._add(GateOp("cz", (q0, q1)))

    def swap(self, q0: int, q1: int) -> "Circuit":
        return self._add(GateOp("swap", (q0, q1)))

    def crx(self, control: int, target: int, param: str | float) -> "Circuit":
        return self._add(GateOp("crx", (control, target), param))

    def cry(self, control: int, target: int, param: str | float) -> "Circuit":
        return self._add(GateOp("cry", (control, target), param))

    def crz(self, control: int, target: int, param: str | float) -> "Circuit":
        return self._add(GateOp("crz", (control, target), param))

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        return len(self.ops)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """All distinct string parameter names referenced in the circuit, in first-use order."""
        seen: dict[str, None] = {}
        for op in self.ops:
            if isinstance(op.param, str) and op.param not in seen:
                seen[op.param] = None
        return tuple(seen.keys())

    def __repr__(self) -> str:
        body = ", ".join(
            f"{op.name}({','.join(map(str, op.wires))}"
            + (f", {op.param!r}" if op.param is not None else "")
            + ")"
            for op in self.ops
        )
        return f"Circuit(n_qubits={self.n_qubits}, ops=[{body}])"
