# QAX Lab API reference

A condensed reference for the public API. For full signatures, see the
docstrings inside each module.

## Circuit construction

```python
from qax import Circuit

c = (
    Circuit(n_qubits=3)
    .h(0)
    .cx(0, 1)
    .ry(2, "theta")
    .cz(1, 2)
)

c.depth              # number of gates
c.parameter_names    # tuple of symbolic parameter names, in first-use order
```

### Available gate methods

Constant single-qubit: `i`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`.
Parameterized single-qubit: `rx(w, p)`, `ry(w, p)`, `rz(w, p)`, `phase(w, p)`.
Two-qubit: `cx` / `cnot`, `cz`, `swap`, `crx`, `cry`, `crz`.

`p` may be a Python float (baked in as a literal) or a string (treated as
a symbolic parameter slot, filled in by the `params` dict at simulation time).

## Compilation

```python
from qax import compile_circuit
program = compile_circuit(c)         # frozen IR, hashable
program.n_qubits, program.depth, program.parameter_names
```

Compiling once and reusing the `Program` is the right move if you plan to
differentiate, batch, or repeatedly run the same circuit; JAX will cache the
compiled trace keyed on the program.

## Simulation

```python
from qax import simulate, simulate_scan
import jax.numpy as jnp

state = simulate(c, params={"theta": jnp.array(0.1)})
state = simulate(c, params={"theta": jnp.array(0.1)}, jit=False)
state = simulate_scan(c, params={"theta": jnp.array(0.1)})
```

Both return a complex statevector of shape `(2**n_qubits,)`. `simulate` is
fast (tensor-contraction kernel). `simulate_scan` is slower but goes through
`jax.lax.scan` over stacked full-system unitaries — useful for benchmarks
and didactic walkthroughs.

## Observables

```python
from qax import PauliHamiltonian, expectation

h = PauliHamiltonian([("ZZ", 1.0), ("XX", -0.5)])
energy = expectation(state, h)        # real-valued JAX scalar
```

Pauli strings use `I`, `X`, `Y`, `Z`. Order in the string matches qubit
order: `pauli[0]` acts on qubit 0.

## Sampling

```python
from qax import sample_indices, sample_counts
import jax

key = jax.random.PRNGKey(0)
idx = sample_indices(key, state, n_shots=1024)              # JAX array
counts = sample_counts(key, state, n_qubits=3, n_shots=1024)  # dict
```

`sample_indices` is JIT-compiled with `n_shots` static; safe to wrap in
`vmap`. `sample_counts` is a Python-side convenience.

## Optimizers

```python
from qax.optim import adam, sgd

opt = adam(learning_rate=0.05)
opt_state = opt.init(params)

@jax.jit
def train_step(params, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = opt.apply_updates(params, updates)
    return params, opt_state, loss
```

`sgd(learning_rate, momentum=0.0)` and `adam(learning_rate, b1, b2, eps)` are
pure functions returning an `Optimizer` record. State is a registered PyTree.

## Metrics

```python
from qax import fidelity, infidelity, inner_product
from qax.metrics import l2_distance, trace_distance_pure
```

## Noise

```python
from qax import NoiseChannel, apply_channel
import jax

key = jax.random.PRNGKey(0)
ch = NoiseChannel(kind="depolarizing", wire=0, probability=0.1)
new_state = apply_channel(state, n_qubits, ch, key)
```

To estimate mixed-state expectations, wrap the trajectory in `jax.vmap` over
a batch of PRNG keys and average.

## Parameter-shift gradients

```python
from qax.grad_utils import parameter_shift_grad

shift_grad = parameter_shift_grad(loss_fn, params, name="theta")
```

A direct comparison to `jax.grad(loss_fn)(params)["theta"]`; useful for
validation and as a pedagogical reference.

## QASM export

```python
from qax import to_qasm
qasm_text = to_qasm(circuit, params={"theta": 0.1})
```

One-way export of the supported gate set into OpenQASM 2.0 text.
