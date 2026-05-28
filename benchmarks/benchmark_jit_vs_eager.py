"""Benchmark: eager Python loop vs JIT-compiled simulation.

Times a moderately deep circuit on 8 qubits under three regimes:

    1. eager mode (no JIT)
    2. JIT first call (includes compilation)
    3. JIT cached call (the fair number for repeated runs)

Run with::

    uv run python benchmarks/benchmark_jit_vs_eager.py
"""

from __future__ import annotations

import time

import jax

from qax import Circuit, simulate


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
    # Warm up the device once.
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
    n_qubits = 8
    depth = 5
    circuit, params = build_circuit(n_qubits, depth)
    n_ops = len(circuit.ops)
    print(f"Circuit: {n_qubits} qubits, {n_ops} gates\n")

    # Eager mode: rebuild the simulate call each time so we measure no JIT.
    def run_eager(params):
        return simulate(circuit, params, jit=False)

    eager = time_call(run_eager, params, repeats=3)
    print(f"Eager simulation:        {eager * 1e3:8.2f} ms / run")

    # JIT: include first-call compilation time, then cached calls.
    jit_run = jax.jit(lambda p: simulate(circuit, p))
    t0 = time.perf_counter()
    first = jit_run(params)
    jax.block_until_ready(first)
    first_call = time.perf_counter() - t0
    print(f"JIT first call:          {first_call * 1e3:8.2f} ms  (includes compile)")

    cached = time_call(jit_run, params, repeats=10)
    print(f"JIT cached call:         {cached * 1e3:8.2f} ms / run")

    speedup = eager / cached if cached > 0 else float("inf")
    print(f"\nJIT speedup over eager:  {speedup:.1f}x")


if __name__ == "__main__":
    main()
