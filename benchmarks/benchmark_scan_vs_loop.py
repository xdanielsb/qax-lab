"""Benchmark: jax.lax.scan vs the default per-gate JIT loop.

Both backends produce the same final statevector; the scan path is mostly
of interest as a compiled-loop demonstration.

Run with::

    uv run python benchmarks/benchmark_scan_vs_loop.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from qax import Circuit, simulate, simulate_scan


def build_circuit(n_qubits: int, depth: int) -> tuple[Circuit, dict]:
    c = Circuit(n_qubits)
    for layer in range(depth):
        for q in range(n_qubits):
            c = c.ry(q, f"y_{layer}_{q}").rz(q, f"z_{layer}_{q}")
        for q in range(n_qubits - 1):
            c = c.cx(q, q + 1)
    key = jax.random.PRNGKey(0)
    names = c.parameter_names
    init = 0.1 * jax.random.normal(key, shape=(len(names),))
    params = {n: init[i] for i, n in enumerate(names)}
    return c, params


def time_call(fn, *args, repeats: int = 5) -> float:
    out = fn(*args)
    jax.block_until_ready(out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return min(times)


def main() -> None:
    n_qubits = 6
    depth = 5
    circuit, params = build_circuit(n_qubits, depth)
    print(f"Circuit: {n_qubits} qubits, {len(circuit.ops)} gates\n")

    jit_default = jax.jit(lambda p: simulate(circuit, p))
    jit_scan = jax.jit(lambda p: simulate_scan(circuit, p))

    # Verify they agree.
    s_default = jit_default(params)
    s_scan = jit_scan(params)
    diff = float(jnp.max(jnp.abs(s_default - s_scan)))
    print(f"Max |default - scan|:    {diff:.2e}")

    default_t = time_call(jit_default, params, repeats=10)
    scan_t = time_call(jit_scan, params, repeats=10)

    print(f"\nJIT default (cached):    {default_t * 1e3:8.2f} ms / run")
    print(f"JIT scan    (cached):    {scan_t * 1e3:8.2f} ms / run")


if __name__ == "__main__":
    main()
