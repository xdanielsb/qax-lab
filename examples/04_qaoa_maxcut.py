"""QAOA for MaxCut on a 4-node cycle graph.

For a graph G=(V,E) the cost Hamiltonian is

    H_C = sum_{(i,j) in E} 1/2 * (1 - Z_i Z_j)

and the mixer is

    H_M = sum_i X_i

QAOA prepares the state

    |psi(gamma, beta)> = U_M(beta_p) U_C(gamma_p) ... U_M(beta_1) U_C(gamma_1) |+>^n

and optimizes ``<psi|H_C|psi>``. We use a small p=2 ansatz.

Also demonstrates :func:`jax.vmap` across multiple random initializations.

Run with::

    uv run python examples/04_qaoa_maxcut.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qax import Circuit, PauliHamiltonian, expectation, simulate
from qax.optim import adam


def maxcut_hamiltonian(n: int, edges: list[tuple[int, int]]) -> PauliHamiltonian:
    """Build sum_{(i,j) in E} 1/2 * (1 - Z_i Z_j)."""
    terms: list[tuple[str, float]] = []
    for (i, j) in edges:
        # Identity term contributes a constant 0.5 per edge.
        terms.append(("I" * n, 0.5))
        # The Z_i Z_j term contributes -0.5.
        pauli = ["I"] * n
        pauli[i] = "Z"
        pauli[j] = "Z"
        terms.append(("".join(pauli), -0.5))
    return PauliHamiltonian(terms)


def build_qaoa_circuit(n: int, edges: list[tuple[int, int]], p: int) -> Circuit:
    c = Circuit(n)
    # initial Hadamard layer
    for q in range(n):
        c = c.h(q)
    for layer in range(p):
        # cost layer: exp(-i * gamma * H_C). Each ZZ rotation is implemented as
        # CX(i,j) RZ(2*gamma, j) CX(i,j) up to a global phase that is constant.
        for (i, j) in edges:
            c = c.cx(i, j).rz(j, f"gamma_{layer}").cx(i, j)
        # mixer layer: exp(-i * beta * sum_q X_q) = prod_q RX(2*beta, q).
        for q in range(n):
            c = c.rx(q, f"beta_{layer}")
    return c


def main() -> None:
    n = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    p = 2

    hc = maxcut_hamiltonian(n, edges)
    ansatz = build_qaoa_circuit(n, edges, p=p)

    @jax.jit
    def energy(params):
        state = simulate(ansatz, params)
        return expectation(state, hc)

    @jax.jit
    def loss(params):
        # We want to maximize <H_C>, equivalently minimize -<H_C>.
        return -energy(params)

    # ---- single-restart Adam ----
    key = jax.random.PRNGKey(0)
    names = ansatz.parameter_names
    init = 0.1 * jax.random.normal(key, shape=(len(names),))
    params = {n_: init[i] for i, n_ in enumerate(names)}

    opt = adam(learning_rate=0.05)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        val, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = opt.apply_updates(params, updates)
        return params, opt_state, val

    print("=" * 60)
    print(f"QAOA MaxCut on {n}-node cycle  (p={p})")
    print("=" * 60)
    print(f"Edges: {edges}")
    print(f"Maximum possible cut: {len(edges)}")
    print(f"\nInitial <H_C>: {float(energy(params)):.4f}")

    n_steps = 300
    for s in range(n_steps):
        params, opt_state, val = step(params, opt_state)
        if (s + 1) % 30 == 0 or s == 0:
            print(f"  step {s + 1:4d}  <H_C>={float(energy(params)):.4f}")

    print(f"\nFinal single-restart <H_C>: {float(energy(params)):.4f}")

    # ---- vmap across many random restarts ----
    print("\n" + "-" * 60)
    print("vmap demo: evaluate energy across 64 random parameter sets in one call")
    print("-" * 60)
    keys = jax.random.split(jax.random.PRNGKey(1), 64)

    def random_params(k):
        v = jax.random.uniform(k, shape=(len(names),), minval=0.0, maxval=jnp.pi)
        return {n_: v[i] for i, n_ in enumerate(names)}

    batch_params = jax.vmap(random_params)(keys)
    batch_energy = jax.jit(jax.vmap(energy))(batch_params)
    print(f"Mean random <H_C>: {float(jnp.mean(batch_energy)):.4f}")
    print(f"Best random <H_C>: {float(jnp.max(batch_energy)):.4f}")


if __name__ == "__main__":
    main()
