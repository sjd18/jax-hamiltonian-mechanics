"""Order of accuracy of each integrator on an eccentric Kepler orbit.

Also shows ``jax.vmap`` over the step size: because ``dt`` can be a traced
array, a single compiled rollout runs a whole batch of step sizes at once.
Saves ``assets/convergence.png``.

    python examples/convergence.py
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import _style  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import hamjax as hj  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def main():
    state, masses, period = hj.kepler_two_body(eccentricity=0.5)
    H = hj.NBody(masses)

    # Integrate exactly one period with n = 64 … 4096 steps and measure how
    # far each integrator lands from the starting point.
    counts = 2 ** np.arange(6, 13)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, integrator, order in [
        ("Velocity Verlet", hj.VelocityVerlet(), 2),
        ("RK4", hj.RK4(), 4),
        ("Yoshida 4", hj.Yoshida4(), 4),
    ]:
        errs = []
        for n in counts:
            tr = hj.simulate(
                H, state, dt=period / n, n_steps=int(n), integrator=integrator, save_every=int(n)
            )
            errs.append(float(jnp.max(jnp.abs(tr.q[-1] - state.q))))
        errs = np.array(errs)
        dts = period / counts
        slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        ax.loglog(
            dts,
            errs,
            "o-",
            color=_style.COLORS[name],
            ms=8,
            mec=_style.SURFACE,
            mew=2,
            label=f"{name} (slope {slope:.2f})",
        )
        print(f"{name:16s} measured order {slope:.2f} (expected {order})")

    ax.set_title("Global position error after one Kepler orbit (e = 0.5)")
    ax.set_xlabel("time step dt")
    ax.set_ylabel("max |q(T) − q(0)|")
    ax.legend(loc="lower right")
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "convergence.png")
    print("saved", ASSETS / "convergence.png")

    # vmap over dt: one compiled kernel, a batch of step sizes.
    dts = period / jnp.array([500.0, 1000.0, 2000.0])
    final_q = jax.vmap(lambda dt: hj.simulate(H, state, dt=dt, n_steps=500, save_every=500).q[-1])(dts)
    print("vmapped over dt -> final positions batch shape", final_q.shape)


if __name__ == "__main__":
    main()
