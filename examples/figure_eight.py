"""The figure-eight three-body choreography, integrated for 1,000 periods.

Compares velocity Verlet (symplectic, 2nd order), Yoshida-4 (symplectic, 4th
order) and RK4 (non-symplectic, 4th order) at the same step size, and saves
``assets/figure_eight.png``.

    python examples/figure_eight.py
"""

import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import _style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import hamjax as hj  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def main(periods: int = 1000, steps_per_period: int = 100):
    state, masses, period = hj.figure_eight()
    H = hj.NBody(masses)
    dt = period / steps_per_period
    n = periods * steps_per_period

    runs = {}
    for name, integrator in [
        ("Velocity Verlet", hj.VelocityVerlet()),
        ("Yoshida 4", hj.Yoshida4()),
        ("RK4", hj.RK4()),
    ]:
        hj.simulate(H, state, dt=dt, n_steps=n, integrator=integrator, save_every=steps_per_period)  # compile
        t0 = time.perf_counter()
        tr = hj.simulate(H, state, dt=dt, n_steps=n, integrator=integrator, save_every=steps_per_period)
        tr.q.block_until_ready()
        elapsed = time.perf_counter() - t0
        err = hj.relative_energy_error(H, tr)
        runs[name] = (tr, err)
        print(f"{name:16s} {n:>7d} steps in {elapsed * 1e3:7.1f} ms   max |ΔE/E| = {float(err.max()):.2e}")

    orbit = hj.simulate(H, state, dt=period / 400, n_steps=400)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.35]})
    # All three bodies trace the same curve; show it once, plus the bodies at t = 0.
    ax0.plot(orbit.q[:, 0, 0], orbit.q[:, 0, 1], color=_style.TEXT_2, lw=1.5)
    for i, c in enumerate([_style.BLUE, _style.ORANGE, _style.AQUA]):
        ax0.plot(*orbit.q[0, i], "o", ms=10, color=c, mec=_style.SURFACE, mew=2, zorder=3)
    ax0.set_aspect("equal")
    ax0.set_title("Figure-eight choreography (one period)")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    for name, (tr, err) in runs.items():
        ax1.loglog(tr.t[1:] / period, err[1:], color=_style.COLORS[name], label=name)
        ax1.annotate(
            name,
            (tr.t[-1] / period, err[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=_style.TEXT,
            fontsize=9,
        )
    ax1.set_title(f"Relative energy error, dt = T/{steps_per_period}")
    ax1.set_xlabel("time [periods]")
    ax1.set_ylabel("|H(t) − H(0)| / |H(0)|")
    ax1.legend(loc="upper left")
    ax1.set_xlim(right=periods * 4)
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "figure_eight.png")
    print("saved", ASSETS / "figure_eight.png")


if __name__ == "__main__":
    main()
