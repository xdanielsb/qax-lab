"""Minimal OpenQASM 2.0 export for the supported gate set.

This is a *one-way* export: we render ``Circuit`` objects to QASM text but
do not attempt to parse QASM back in. The goal is just to provide a portable
hand-off format so users can paste circuits into other tools.
"""

from __future__ import annotations

from .circuit import Circuit

_GATE_RENDERERS: dict[str, str] = {
    "i": "id",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "s": "s",
    "sdg": "sdg",
    "t": "t",
    "tdg": "tdg",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "phase": "u1",
    "cnot": "cx",
    "cx": "cx",
    "cz": "cz",
    "swap": "swap",
    "crx": "crx",
    "cry": "cry",
    "crz": "crz",
}


def to_qasm(circuit: Circuit, params: dict | None = None) -> str:
    """Render ``circuit`` as OpenQASM 2.0 text.

    Parameterized gates resolve string parameter names through ``params``;
    any unresolved symbolic parameter is rendered as the bare name (which
    keeps the output a valid *template*, but not strictly valid QASM).
    """
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{circuit.n_qubits}];",
    ]
    for op in circuit.ops:
        gate = _GATE_RENDERERS.get(op.name)
        if gate is None:
            raise ValueError(f"No QASM rendering for gate {op.name!r}.")
        param = op.param
        if isinstance(param, str):
            if params and param in params:
                param_value = float(params[param])
                head = f"{gate}({param_value:.10g})"
            else:
                head = f"{gate}({param})"
        elif param is not None:
            head = f"{gate}({float(param):.10g})"
        else:
            head = gate
        wires = ", ".join(f"q[{w}]" for w in op.wires)
        lines.append(f"{head} {wires};")
    return "\n".join(lines) + "\n"
