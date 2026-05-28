"""GHZ state demo.

Build the three-qubit GHZ state ``(|000> + |111>) / sqrt(2)``. Run with::

    uv run python examples/02_ghz_state.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qax import Circuit, probabilities, sample_counts, simulate


def main() -> None:
    circuit = Circuit(n_qubits=3).h(0).cx(0, 1).cx(1, 2)
    state = simulate(circuit)

    print("=" * 60)
    print("GHZ state demo")
    print("=" * 60)
    print(f"\nCircuit: {circuit}")
    print("\nNon-zero amplitudes:")
    arr = jnp.asarray(state)
    for i, amp in enumerate(arr):
        if abs(complex(amp)) > 1e-6:
            print(f"  |{i:03b}>: {complex(amp):.4f}")

    probs = probabilities(state)
    print("\nNon-zero probabilities:")
    for i, p in enumerate(jnp.asarray(probs)):
        if float(p) > 1e-6:
            print(f"  |{i:03b}>: {float(p):.4f}")

    key = jax.random.PRNGKey(0)
    counts = sample_counts(key, state, n_qubits=3, n_shots=2048)
    print("\nMeasurement counts (2048 shots):")
    for bitstring, cnt in counts.items():
        print(f"  |{bitstring}>: {cnt}")


if __name__ == "__main__":
    main()
