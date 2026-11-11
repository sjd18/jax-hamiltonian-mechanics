"""A self-gravitating Plummer star cluster.

``N`` softened bodies, O(N²) forces from ``jax.grad`` of the pair potential,
a jitted ``lax.scan`` rollout, and conservation diagnostics. Saves
``assets/plummer.png``.

    python examples/plummer.py --n 1024 --steps 4000
"""

import argparse
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import _style  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import hamjax as hj  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def main(n: int, steps: int, dt: float, softening: float):
    state, masses = hj.plummer_sphere(jax.random.PRNGKey(0), n)
    H = hj.NBody(masses, softening=softening)
    save_every = steps // 200

    run = lambda: hj.simulate(H, state, dt=dt, n_steps=steps, save_every=save_every)  # noqa: E731
    run().q.block_until_ready()  # compile
    t0 = time.perf_counter()
    tr = run()
    tr.q.block_until_ready()
    elapsed = time.perf_counter() - t0
    print(
        f"N={n}: {steps} steps in {elapsed:.2f} s  ({elapsed / steps * 1e3:.2f} ms/step, "
        f"{n * n * steps / elapsed / 1e9:.2f} G pair-interactions/s)"
    )

    dE = hj.relative_energy_error(H, tr)
    P = hj.total_momentum(tr.state)
    L = hj.angular_momentum(tr.state)
    print(f"max |ΔE/E| = {float(dE.max()):.2e}")
    print(f"max |ΔP|   = {float(jnp.abs(P - P[0]).max()):.2e}")
    print(f"max |ΔL|   = {float(jnp.abs(L - L[0]).max()):.2e}")

    fig = plt.figure(figsize=(11, 3.8))
    for k, idx in enumerate([0, len(tr.t) // 2, len(tr.t) - 1]):
        ax = fig.add_subplot(1, 4, k + 1)
        ax.scatter(tr.q[idx, :, 0], tr.q[idx, :, 1], s=2, color=_style.BLUE, alpha=0.6, lw=0)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_title(f"t = {float(tr.t[idx]):.1f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    ax = fig.add_subplot(1, 4, 4)
    ax.semilogy(tr.t[1:], dE[1:], color=_style.BLUE)
    ax.set_title("|ΔE / E|", fontsize=10)
    ax.set_xlabel("time [Hénon units]")
    fig.suptitle(
        f"Plummer sphere, N = {n}, velocity Verlet, ε = {softening}",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=_style.TEXT,
    )
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "plummer.png")
    print("saved", ASSETS / "plummer.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--dt", type=float, default=2.5e-3)
    ap.add_argument("--softening", type=float, default=0.02)
    a = ap.parse_args()
    main(a.n, a.steps, a.dt, a.softening)
