"""Bell state demo.

Build the maximally entangled two-qubit state ``(|00> + |11>) / sqrt(2)``,
print its amplitudes, then sample shots from it. Run with::

    uv run python examples/01_bell_state.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qax import Circuit, probabilities, sample_counts, simulate


def main() -> None:
    circuit = Circuit(n_qubits=2).h(0).cx(0, 1)
    state = simulate(circuit)

    print("=" * 60)
    print("Bell state demo")
    print("=" * 60)
    print(f"\nCircuit: {circuit}")
    print(f"\nStatevector amplitudes:")
    for i, amp in enumerate(jnp.asarray(state)):
        print(f"  |{i:02b}>: {complex(amp):.4f}")

    probs = probabilities(state)
    print(f"\nProbabilities:")
    for i, p in enumerate(jnp.asarray(probs)):
        print(f"  |{i:02b}>: {float(p):.4f}")

    key = jax.random.PRNGKey(0)
    counts = sample_counts(key, state, n_qubits=2, n_shots=1024)
    print(f"\nMeasurement counts (1024 shots):")
    for bitstring, cnt in counts.items():
        print(f"  |{bitstring}>: {cnt}")


if __name__ == "__main__":
    main()
