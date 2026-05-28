# QAX Lab

**Differentiable quantum circuits in JAX.**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.30+-orange.svg)](https://github.com/jax-ml/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)

QAX Lab is a from-scratch, JAX-native quantum circuit simulator. It is a
compact demonstration of differentiable scientific computing in JAX —
complex statevectors, parameterized gates, batched experiments, gradient-
based circuit optimization, all in a single dependency-light package.

```python
import jax
import jax.numpy as jnp

from qax import Circuit, PauliHamiltonian, expectation, simulate

circuit = (
    Circuit(n_qubits=3)
    .h(0)
    .cx(0, 1)
    .ry(2, "theta")
    .cz(1, 2)
)

hamiltonian = PauliHamiltonian([
    ("ZII", 1.0),
    ("IZZ", -0.5),
])

params = {"theta": jnp.array(0.1)}

state = simulate(circuit, params)
energy = expectation(state, hamiltonian)
grads = jax.grad(lambda p: expectation(simulate(circuit, p), hamiltonian))(params)
```

## Why this exists

QAX Lab it combines JAX with complex-valued scientific computing, quantum circuits,
differentiable simulation, batching, benchmarking, and clean package design.

The simulator core has **no dependency on Qiskit, PennyLane, Cirq, or
TensorFlow Quantum**.

## Install

This project uses [uv](https://docs.astral.sh/uv/) by default.

```bash
git clone https://github.com/qax-lab/qax-lab.git
cd qax-lab
uv sync --extra dev
```

Plain pip works too:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Features at a glance

| JAX concept | Where it appears |
|---|---|
| `jax.numpy` | Complex-valued statevectors and gate matrices everywhere. |
| `jax.jit` | `simulate(..., jit=True)` compiles the whole circuit; benchmarks isolate first-call vs cached cost. |
| `jax.grad` / `value_and_grad` | Drives `examples/03_circuit_synthesis.py`, `examples/04_qaoa_maxcut.py`, `examples/05_tiny_vqe.py`. |
| `jax.vmap` | QAOA random restarts and the `benchmark_vmap_batching.py` script. |
| `jax.lax.scan` | `simulate_scan` runs a compiled per-gate loop; benchmarked against the default backend. |
| PyTrees | Parameters live in plain `dict[str, jnp.ndarray]`; the Adam optimizer state is a registered PyTree. |
| `jax.random` | Measurement sampling and Monte Carlo noise channels use explicit PRNG keys. |
| `custom_jvp` analogue | `qax.grad_utils.parameter_shift_grad` shows the textbook parameter-shift rule alongside `jax.grad`. |

## Quick tour

### Build a circuit

```python
from qax import Circuit

circuit = (
    Circuit(n_qubits=4)
    .h(0)
    .cx(0, 1)
    .ry(2, "theta_1")
    .rz(3, "theta_2")
    .cz(2, 3)
)
```

`Circuit` is an immutable dataclass: every gate method returns a brand-new
instance, which makes it safe to hash and pass as a static argument to
`jax.jit`.

### Run the simulation

```python
from qax import simulate
import jax.numpy as jnp

params = {"theta_1": jnp.array(0.25), "theta_2": jnp.array(-0.8)}
state = simulate(circuit, params)
```

`simulate` accepts either a `Circuit` or a pre-compiled `Program`. Compile
once and reuse if you plan to differentiate or batch.

### Differentiate

```python
import jax
from qax import PauliHamiltonian, expectation, simulate

H = PauliHamiltonian([("ZIII", 1.0), ("IIZZ", -0.5)])

@jax.jit
def energy(params):
    return expectation(simulate(circuit, params), H)

grads = jax.grad(energy)(params)
```

### Batch with vmap

```python
keys = jax.random.split(jax.random.PRNGKey(0), 128)
def random_params(k):
    v = jax.random.uniform(k, (len(circuit.parameter_names),), minval=0, maxval=jnp.pi)
    return {n: v[i] for i, n in enumerate(circuit.parameter_names)}

batched_params = jax.vmap(random_params)(keys)
batched_energy = jax.jit(jax.vmap(energy))(batched_params)
```

### Sample shots

```python
from qax import sample_counts

key = jax.random.PRNGKey(0)
counts = sample_counts(key, state, n_qubits=4, n_shots=1024)
```

## Examples

| File | What it shows |
|---|---|
| [`examples/01_bell_state.py`](examples/01_bell_state.py) | Build `(\|00⟩+\|11⟩)/√2`, print amplitudes, sample shots. |
| [`examples/02_ghz_state.py`](examples/02_ghz_state.py) | Build `(\|000⟩+\|111⟩)/√2`. |
| [`examples/03_circuit_synthesis.py`](examples/03_circuit_synthesis.py) | Learn ansatz parameters that produce a GHZ target via `jax.grad` + Adam. |
| [`examples/04_qaoa_maxcut.py`](examples/04_qaoa_maxcut.py) | QAOA on a 4-node cycle graph; demonstrates `vmap` across random restarts. |
| [`examples/05_tiny_vqe.py`](examples/05_tiny_vqe.py) | VQE for `ZI + IZ + 0.5·XX`; compared against the exact ground-state energy. |

Run them with:

```bash
uv run python examples/01_bell_state.py
uv run python examples/02_ghz_state.py
uv run python examples/03_circuit_synthesis.py
uv run python examples/04_qaoa_maxcut.py
uv run python examples/05_tiny_vqe.py
```

## Benchmarks

```bash
uv run python benchmarks/benchmark_jit_vs_eager.py
uv run python benchmarks/benchmark_vmap_batching.py
uv run python benchmarks/benchmark_scan_vs_loop.py
```

Sample output on a developer laptop (CPU; exact numbers depend on hardware):

```text
benchmark_jit_vs_eager.py
-------------------------
Circuit: 8 qubits, 115 gates
Eager simulation:           70.82 ms / run
JIT first call:            613.82 ms  (includes compile)
JIT cached call:             0.30 ms / run
JIT speedup over eager:    236x

benchmark_vmap_batching.py
--------------------------
Circuit: 6 qubits, 51 gates, batch=256
Sequential (256 calls):    985.12 ms
vmap   (256 in one call):    5.58 ms
vmap speedup over loop:    177x

benchmark_scan_vs_loop.py
-------------------------
Circuit: 6 qubits, 85 gates
Max |default - scan|:        1.80e-07
JIT default (cached):        0.17 ms / run
JIT scan    (cached):        1.23 ms / run
```

The "first call" timing includes XLA compilation and is the fair number
**only for one-shot evaluations**. For training loops and other repeated
workloads the cached call is the right comparison. The `scan` path is
slower because it explicitly materializes full-system unitaries; the
default backend contracts gates directly against the statevector and
keeps memory at `O(2^n)` per gate.

## Tests

```bash
uv run pytest -q
```

Coverage spans gate matrices, statevector correctness, parameterized
simulation, observables, sampling distributions, gradients (including a
parameter-shift sanity check), and the optimizers. Bell, GHZ, and Pauli
expectation values are checked against their analytic values.

## Project layout

```text
qax-lab/
├── qax/                         # core package
│   ├── __init__.py
│   ├── circuit.py               # immutable Circuit / GateOp
│   ├── gates.py                 # I X Y Z H S T RX RY RZ CNOT CZ SWAP …
│   ├── grad_utils.py            # parameter-shift rule (educational)
│   ├── metrics.py               # fidelity / infidelity / trace distance
│   ├── noise.py                 # Monte Carlo single-qubit noise
│   ├── observables.py           # PauliHamiltonian + expectation
│   ├── optim.py                 # hand-rolled SGD + Adam
│   ├── program.py               # compiled, JIT-friendly Program
│   ├── qasm.py                  # one-way OpenQASM 2.0 export
│   ├── sampling.py              # explicit-PRNG measurement sampling
│   ├── simulate.py              # tensor-contraction + lax.scan backends
│   ├── state.py                 # zero_state / basis_state / helpers
│   └── typing.py                # shared type aliases
├── examples/01..05_*.py
├── benchmarks/benchmark_*.py
├── tests/test_*.py
├── docs/architecture.md
├── docs/api.md
└── pyproject.toml
```

## Design notes

- **Big-endian bit ordering.** The leftmost qubit in a multi-qubit gate is the
  most significant bit of the basis-state index, i.e. `|q0 q1⟩` ↦ index
  `2*q0 + q1`. This matches the textbook convention used throughout the
  documentation.
- **`Circuit` is immutable.** All builder methods return new instances. The
  type is `frozen=True` dataclass, hashable, and safe as a static argument to
  `jax.jit`.
- **Parameters are PyTrees.** They live in `dict[str, jnp.ndarray]` (or any
  pytree the user prefers). Names referenced in the circuit must appear in
  the dict at simulation time, or you get a clear `KeyError` before entering
  the JIT-compiled path.
- **Tensor contraction, not full-system matrices.** `simulate` applies each
  gate by reshaping the statevector and using `tensordot`/`moveaxis`. Memory
  stays at `O(2^n)`. A pedagogically simpler `simulate_scan` path *is*
  available, which stacks full-system unitaries and scans across them.
- **Noiseless core stays pure.** Noise lives in `qax.noise` and is implemented
  as randomized Monte Carlo trajectories. To get mixed-state expectations,
  `vmap` the simulation across a batch of PRNG keys and average.

See [`docs/architecture.md`](docs/architecture.md) for a fuller treatment.

## Roadmap

Implemented:

- Phase 1 — core simulator with all required gates, Bell + GHZ demos, tests.
- Phase 2 — differentiable parameters and fidelity metric; circuit synthesis
  example reaches >0.99 fidelity.
- Phase 3 — Pauli observables, sampling, hand-rolled SGD + Adam, tiny VQE.
- Phase 4 — `jax.vmap` batching, `lax.scan` backend, three benchmarks.
- Phase 5 — Monte Carlo noise channels, OpenQASM export, parameter-shift
  gradients as an educational comparison against `jax.grad`.

Possible next steps:

- Density-matrix backend for exact small-system noise.
- A small Streamlit front end that visualizes circuit synthesis live.
- GPU-aware benchmarks once the user adds `jax[cuda12]` to their install.

## License

MIT.
