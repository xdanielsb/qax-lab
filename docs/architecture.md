# QAX Lab architecture

This document explains how the pieces fit together: what gets compiled, what
gets differentiated, and why the data structures look the way they do.

## Layered design

```
                  ┌─────────────────────────┐
   user code  →   │       Circuit           │   immutable builder
                  └────────────┬────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │  compile_circuit (pure) │   freezes wires + param slots
                  └────────────┬────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │        Program          │   frozen, hashable, static
                  └────────────┬────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │  simulate / simulate_   │   pure JAX function:
                  │  scan / expectation     │   safe under jit/grad/vmap
                  └────────────┬────────────┘
                               ▼
                            statevector
```

The split between **builder** (`Circuit`), **compiled IR** (`Program`), and
**runtime** (`simulate`) is the only structural choice that really matters.
It is the same pattern XLA uses: user code constructs a value, the value is
lowered into a frozen representation, and runtime functions consume the
representation.

`Circuit` is convenient but heterogenous (its `param` field is a `str | float
| None`). `Program` normalizes that into a uniform `ProgramOp` shape so the
backend doesn't have to branch on Python types every step.

## Why `Circuit` is immutable

`@dataclass(frozen=True)` plus a tuple of frozen ops gives us three useful
properties for free:

1. **Hashable**, so we can pass it as a static argument to `jax.jit`. JAX
   caches compiled traces keyed on the hash of the static args; an immutable
   circuit means equal circuits hit the cache.
2. **Safe to share**. Two callers can hold references to the same `Circuit`
   without worrying that one of them might mutate it.
3. **Cheap to reason about in tests**. Equality is structural, so we can
   compare two circuits with `==`.

## Why parameters are not part of the program

`Program` deliberately stores parameter *names*, not values. The values come
in as a separate PyTree at `simulate(...)`. This is what enables three things
at once:

- **`jax.grad` works** — the parameters are JAX arrays, so `grad` can
  differentiate the function `params → expectation`.
- **`jax.jit` caches well** — the `Program` is static (compile cache key);
  the `params` PyTree is dynamic (no re-trace on each new gradient step).
- **`jax.vmap` works** — broadcasting over the leading axis of `params` is
  the natural way to batch over many parameter sets at once.

If we baked parameter values into `Program`, every gradient step would
rebuild the program and force a re-trace.

## The tensor-contraction kernel

Rather than building `2^n × 2^n` matrices and multiplying them into the
state, we reshape the statevector to a tensor of shape `(2,) * n` and apply
each gate with `jnp.tensordot` + `jnp.moveaxis`:

```python
state_t = state.reshape((2,) * n)
gate_t = gate.reshape((2,) * k + (2,) * k)
contracted = jnp.tensordot(gate_t, state_t, axes=(list(range(k, 2*k)), list(wires)))
out = jnp.moveaxis(contracted, list(range(k)), list(wires))
```

That keeps the working memory at `O(2^n)` instead of `O(4^n)`. For `n = 10`
qubits that is the difference between `~8 KB` and `~8 MB` per gate — enough
to make a real benchmark difference.

`simulate_scan`, by contrast, exists as a *pedagogical* path: it stacks
`depth` full-system unitaries and feeds them into `jax.lax.scan`. It is
slower because of the `O(4^n)` matrices but it gives you the cleanest
possible reading of "compiled gate loop" from JAX's perspective.

## Observables

A `PauliHamiltonian` is a tuple of `(pauli_string, coeff)` pairs. We do **not**
build the full operator; instead `expectation(state, H)` walks the terms and
computes each `<psi|P|psi>` by applying `P` to `state` (reusing the same
tensor-contraction kernel) and taking `vdot(state, P|psi>).real`.

The cost is linear in the number of terms and the inner Pauli applications
fuse cleanly under `jit`.

## Sampling

We split sampling into a JAX-friendly half and a Python display half. The
JAX half (`sample_indices`) is jitted with `n_shots` as a static argument
and returns an integer array. The Python half (`counts_from_indices`)
converts that array to a `{bitstring: count}` dict for pretty printing.

Keeping the two separate means the hot path stays compatible with `jit` and
`vmap` while you still get a humane representation for printing.

## Optimizers

`sgd` and `adam` are pure functions returning `Optimizer` records with three
callables — `init`, `update`, `apply_updates` — that operate on PyTrees of
arrays. There is no global mutable state.

For Adam we register `AdamState` with `jax.tree_util` so the optimizer state
can be passed through `jit`, `scan`, and `vmap` like any other PyTree.

## Noise

The noise channels are intentionally not part of the main backend. To get
mixed-state averages, run the simulator under `vmap` over a batch of PRNG
keys and average the per-trajectory results. This keeps the noiseless core
trivially auditable and lets the same `simulate` function serve both noisy
and noiseless workflows.

## Quick reference: file responsibilities

| File | Owns |
|---|---|
| `qax/circuit.py` | `Circuit`, `GateOp` — builder and immutable IR. |
| `qax/program.py` | `Program`, `ProgramOp`, `compile_circuit`. Validation. |
| `qax/gates.py` | Gate matrix constructors, gate registry, dispatch. |
| `qax/state.py` | `zero_state`, `basis_state`, probabilities, normalize. |
| `qax/simulate.py` | Tensor-contraction kernel; `simulate` (jit) + `simulate_scan` (lax.scan). |
| `qax/observables.py` | `PauliHamiltonian`, `expectation`, vectorized variant. |
| `qax/sampling.py` | `sample_indices` (jit, `jax.random.categorical`) + display helpers. |
| `qax/metrics.py` | `fidelity`, `infidelity`, `trace_distance_pure`, `l2_distance`. |
| `qax/optim.py` | `sgd`, `adam`, plus the small `Optimizer` namedtuple-like record. |
| `qax/noise.py` | Single-qubit Monte Carlo bit-flip / phase-flip / depolarizing. |
| `qax/grad_utils.py` | `parameter_shift_grad` as an educational comparison against `jax.grad`. |
| `qax/qasm.py` | One-way OpenQASM 2.0 export. |
| `qax/typing.py` | Shared aliases (`Array`, `ParamTree`, `PRNGKey`). |
