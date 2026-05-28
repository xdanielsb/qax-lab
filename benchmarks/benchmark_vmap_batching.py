"""Benchmark: vmap-batched simulation vs a sequential Python loop.

Same circuit, but evaluated across many parameter sets. ``vmap`` should
amortize the per-call overhead by a large factor.

Run with::

    uv run python benchmarks/benchmark_vmap_batching.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from qax import Circuit, PauliHamiltonian, expectation, simulate


def build_circuit(n_qubits: int, depth: int) -> Circuit:
    c = Circuit(n_qubits)
    for layer in range(depth):
        for q in range(n_qubits):
            c = c.ry(q, f"y_{layer}_{q}").rz(q, f"z_{layer}_{q}")
        for q in range(n_qubits - 1):
            c = c.cx(q, q + 1)
    return c


def main() -> None:
    n_qubits = 6
    depth = 3
    batch_size = 256

    circuit = build_circuit(n_qubits, depth)
    hamiltonian = PauliHamiltonian([("Z" + "I" * (n_qubits - 1), 1.0)])

    names = circuit.parameter_names
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    def make_params(k):
        v = jax.random.uniform(k, shape=(len(names),), minval=0.0, maxval=jnp.pi)
        return {n: v[i] for i, n in enumerate(names)}

    batched_params = jax.vmap(make_params)(keys)
    single_params = jax.tree.map(lambda x: x[0], batched_params)

    @jax.jit
    def one_energy(p):
        return expectation(simulate(circuit, p), hamiltonian)

    batched_energy = jax.jit(jax.vmap(one_energy))

    # Warm up both paths.
    jax.block_until_ready(one_energy(single_params))
    jax.block_until_ready(batched_energy(batched_params))

    print(f"Circuit: {n_qubits} qubits, {len(circuit.ops)} gates, batch={batch_size}\n")

    # Sequential loop.
    t0 = time.perf_counter()
    results = []
    for i in range(batch_size):
        ps = jax.tree.map(lambda x, i=i: x[i], batched_params)
        results.append(one_energy(ps))
    jax.block_until_ready(jnp.stack(results))
    sequential = time.perf_counter() - t0
    print(f"Sequential ({batch_size} calls):      {sequential * 1e3:8.2f} ms")

    # vmap.
    t0 = time.perf_counter()
    vmap_out = batched_energy(batched_params)
    jax.block_until_ready(vmap_out)
    vmap_time = time.perf_counter() - t0
    print(f"vmap   ({batch_size} in one call):  {vmap_time * 1e3:8.2f} ms")

    speedup = sequential / vmap_time if vmap_time > 0 else float("inf")
    print(f"\nvmap speedup over loop:  {speedup:.1f}x")


if __name__ == "__main__":
    main()
