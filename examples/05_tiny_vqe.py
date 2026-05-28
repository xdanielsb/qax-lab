"""A tiny variational quantum eigensolver.

We minimize ``<psi(theta)|H|psi(theta)>`` for a small two-qubit Hamiltonian
using a hardware-efficient ansatz and the hand-rolled Adam optimizer. The
ground-state energy of ``H = ZI + IZ + 0.5 * XX`` can be computed exactly
by diagonalizing the 4x4 matrix; we report that as the ground-truth target.

Run with::

    uv run python examples/05_tiny_vqe.py
"""

from __future__ import annotations

import jax
import numpy as np

from qax import Circuit, PauliHamiltonian, expectation, simulate
from qax.optim import adam


def build_dense_hamiltonian() -> np.ndarray:
    """Return the dense 4x4 matrix for ZI + IZ + 0.5 XX."""
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = np.kron(Z, I) + np.kron(I, Z) + 0.5 * np.kron(X, X)
    return H


def hardware_efficient(n: int, depth: int) -> Circuit:
    c = Circuit(n)
    for layer in range(depth):
        for q in range(n):
            c = c.ry(q, f"y_{layer}_{q}").rz(q, f"z_{layer}_{q}")
        for q in range(n - 1):
            c = c.cx(q, q + 1)
    # final rotation layer
    for q in range(n):
        c = c.ry(q, f"y_final_{q}")
    return c


def main() -> None:
    hamiltonian = PauliHamiltonian([("ZI", 1.0), ("IZ", 1.0), ("XX", 0.5)])
    ansatz = hardware_efficient(n=2, depth=2)

    # Ground truth
    H = build_dense_hamiltonian()
    eigvals = np.linalg.eigvalsh(H)
    ground_truth = float(eigvals.min())

    key = jax.random.PRNGKey(0)
    names = ansatz.parameter_names
    init = 0.3 * jax.random.normal(key, shape=(len(names),))
    params = {n_: init[i] for i, n_ in enumerate(names)}

    @jax.jit
    def energy(params):
        return expectation(simulate(ansatz, params), hamiltonian)

    opt = adam(learning_rate=0.05)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        val, grads = jax.value_and_grad(energy)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = opt.apply_updates(params, updates)
        return params, opt_state, val

    print("=" * 60)
    print("Tiny VQE: H = ZI + IZ + 0.5 * XX")
    print("=" * 60)
    print(f"Exact ground-state energy: {ground_truth:.6f}")
    print("\nAnsatz depth: 2 layers (12 parameters)")
    print(f"Initial energy: {float(energy(params)):.6f}\n")

    n_steps = 500
    history = []
    for s in range(n_steps):
        params, opt_state, val = step(params, opt_state)
        history.append(float(val))
        if (s + 1) % 50 == 0 or s == 0:
            err = float(val) - ground_truth
            print(f"  step {s + 1:4d}  E={float(val):+.6f}  err={err:+.2e}")

    final_energy = float(energy(params))
    print(f"\nFinal energy:        {final_energy:+.6f}")
    print(f"Exact ground energy: {ground_truth:+.6f}")
    print(f"Absolute error:      {abs(final_energy - ground_truth):.2e}")


if __name__ == "__main__":
    main()
