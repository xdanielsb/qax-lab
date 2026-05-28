"""Circuit synthesis demo.

Let JAX learn parameters of an ansatz so that it produces a target GHZ state.
This exercises :func:`jax.grad` end-to-end through the simulator.

Run with::

    uv run python examples/03_circuit_synthesis.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qax import Circuit, fidelity, simulate
from qax.optim import adam


def ghz_state(n_qubits: int) -> jnp.ndarray:
    target = simulate(
        Circuit(n_qubits).h(0).cx(0, 1).cx(1, 2)
    )
    return target


def build_ansatz() -> Circuit:
    return (
        Circuit(3)
        .ry(0, "a").ry(1, "b").ry(2, "c")
        .cx(0, 1).cx(1, 2)
        .rz(0, "d").rz(1, "e").rz(2, "f")
        .cx(0, 1).cx(1, 2)
        .ry(0, "g").ry(1, "h").ry(2, "i")
    )


def main() -> None:
    target = ghz_state(3)
    ansatz = build_ansatz()

    key = jax.random.PRNGKey(0)
    names = ansatz.parameter_names
    init_vals = 0.1 * jax.random.normal(key, shape=(len(names),))
    params = {n: init_vals[i] for i, n in enumerate(names)}

    @jax.jit
    def loss_fn(params):
        state = simulate(ansatz, params)
        return 1.0 - fidelity(state, target)

    initial_loss = float(loss_fn(params))
    initial_fid = 1.0 - initial_loss
    print("=" * 60)
    print("Circuit synthesis: learn GHZ via gradient descent")
    print("=" * 60)
    print(f"\nAnsatz: {ansatz}")
    print(f"\nInitial fidelity: {initial_fid:.4f}")

    opt = adam(learning_rate=0.1)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = opt.apply_updates(params, updates)
        return params, opt_state, loss

    n_steps = 600
    print(f"\nTraining for {n_steps} Adam steps (learning_rate=0.1) ...")
    for step in range(n_steps):
        params, opt_state, loss = train_step(params, opt_state)
        if (step + 1) % 50 == 0 or step == 0:
            fid = 1.0 - float(loss)
            print(f"  step {step + 1:4d}  loss={float(loss):.6f}  fidelity={fid:.6f}")

    final_state = simulate(ansatz, params)
    final_fid = float(fidelity(final_state, target))
    print(f"\nFinal fidelity: {final_fid:.6f}")
    print(f"\nLearned parameters:")
    for k in sorted(params):
        print(f"  {k}: {float(params[k]):+.4f}")


if __name__ == "__main__":
    main()
