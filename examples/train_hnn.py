"""Hamiltonian Neural Networks on the ideal mass-spring (Greydanus et al. 2019).

Three models, all Equinox modules trained with Optax on noisy observations of
``(q, p) -> (q̇, ṗ)``:

* **Baseline** — an MLP that outputs ``(q̇, ṗ)`` directly.
* **HNN** — an MLP that outputs a scalar ``H_θ``; dynamics are ``J∇H_θ``.
* **Separable HNN** — ``H_θ = T_θ(p) + V_θ(q)``, rolled out with the same
  velocity-Verlet integrator as the N-body code (learned force = ``-∇V_θ``).

Each trained model is rolled out from an unseen initial condition and the
*true* energy is tracked along its trajectory. Saves ``assets/hnn.png``.

    python examples/train_hnn.py
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import _style  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import optax  # noqa: E402

import hamjax as hj  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
H_TRUE = hj.HarmonicOscillator(mass=1.0, k=1.0)


def make_dataset(key, n_traj=25, n_obs=30, t_max=3.0, noise=0.1):
    """Trajectories at random energies, sampled with noise (the paper's setup)."""
    k_r, k_th, k_n1, k_n2 = jax.random.split(key, 4)
    radius = jax.random.uniform(k_r, (n_traj,), minval=0.1, maxval=1.0)
    theta = jax.random.uniform(k_th, (n_traj,), maxval=2 * jnp.pi)
    s0 = hj.PhaseSpace((radius * jnp.cos(theta))[:, None], (radius * jnp.sin(theta))[:, None])

    sub = 20
    dt = t_max / (n_obs * sub)
    trajs = jax.vmap(lambda s: hj.simulate(H_TRUE, s, dt=dt, n_steps=n_obs * sub, save_every=sub).state)(s0)
    states = jax.tree.map(lambda x: x[:, 1:].reshape(-1, 1), trajs)
    targets = jax.vmap(lambda s: hj.hamiltonian_vector_field(H_TRUE, s))(states)
    noisy = lambda k, t: jax.tree.map(lambda x: x + noise * jax.random.normal(k, x.shape), t)  # noqa: E731
    return noisy(k_n1, states), noisy(k_n2, targets)


def train(model, states, targets, steps=2000, lr=1e-3):
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(hj.hnn_loss)(model, states, targets)
        updates, opt_state = opt.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), opt_state, loss

    for _ in range(steps):
        model, opt_state, loss = step(model, opt_state)
    return model, float(loss)


@eqx.filter_jit
def rk4_rollout(model, s0, dt, n_steps):
    """RK4 on an arbitrary learned vector field (needed for the baseline)."""
    f = model.time_derivative
    add = lambda a, x, y: jax.tree.map(lambda xi, yi: yi + a * xi, x, y)  # noqa: E731

    def step(z):
        k1 = f(z)
        k2 = f(add(dt / 2, k1, z))
        k3 = f(add(dt / 2, k2, z))
        k4 = f(add(dt, k3, z))
        return jax.tree.map(lambda zi, a, b, c, d: zi + dt / 6 * (a + 2 * b + 2 * c + d), z, k1, k2, k3, k4)

    return hj.rollout(step, s0, n_steps)


def main():
    k_data, k_b, k_h, k_s = jax.random.split(jax.random.PRNGKey(42), 4)
    states, targets = make_dataset(k_data)
    print(f"dataset: {states.q.shape[0]} noisy (state, derivative) pairs")

    models = {
        "Baseline MLP": hj.BaselineMLP(1, key=k_b),
        "HNN": hj.HNN(1, key=k_h),
        "Separable HNN + Verlet": hj.SeparableHNN(1, key=k_s),
    }
    for name in models:
        models[name], loss = train(models[name], states, targets)
        print(f"{name:24s} train loss {loss:.4f}")

    s0 = hj.PhaseSpace(jnp.array([0.75]), jnp.array([0.0]))
    dt, n = 0.05, 1200  # t ∈ [0, 60], 20× the training window
    t = dt * jnp.arange(n + 1)
    truth = hj.simulate(H_TRUE, s0, dt=dt / 10, n_steps=n * 10, save_every=10).state
    rollouts = {
        "Baseline MLP": rk4_rollout(models["Baseline MLP"], s0, dt, n),
        "HNN": rk4_rollout(models["HNN"], s0, dt, n),
        "Separable HNN + Verlet": hj.simulate(models["Separable HNN + Verlet"], s0, dt=dt, n_steps=n).state,
    }
    colors = {"Baseline MLP": _style.ORANGE, "HNN": _style.BLUE, "Separable HNN + Verlet": _style.AQUA}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
    ax0.plot(truth.q[:, 0], truth.p[:, 0], color=_style.TEXT_2, ls="--", lw=1.5, label="Ground truth")
    e_true = H_TRUE(s0)
    for name, traj in rollouts.items():
        ax0.plot(traj.q[:, 0], traj.p[:, 0], color=colors[name], lw=1.5, alpha=0.9, label=name)
        e = jax.vmap(H_TRUE)(traj)
        ax1.plot(t, e, color=colors[name], label=name)
        print(f"{name:24s} true-energy drift at t=60: {float((e[-1] - e_true) / e_true):+.1%}")
    ax1.axhline(float(e_true), color=_style.TEXT_2, ls="--", lw=1.5, label="Ground truth")
    ax1.axvspan(0, 3, color=_style.GRID, alpha=0.6, lw=0)
    ax1.text(
        3.4, ax1.get_ylim()[0], " ← training window", ha="left", va="bottom", fontsize=8, color=_style.TEXT_2
    )

    ax0.set_aspect("equal")
    ax0.set_title("Phase space, t ∈ [0, 60]")
    ax0.set_xlabel("q")
    ax0.set_ylabel("p")
    ax1.set_title("True energy along each learned trajectory")
    ax1.set_xlabel("time")
    ax1.set_ylabel("H(q, p)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=9)
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "hnn.png")
    print("saved", ASSETS / "hnn.png")


if __name__ == "__main__":
    main()
