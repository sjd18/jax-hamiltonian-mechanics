"""Playground: the README Quickstart as a runnable script.

Run it, then edit anything and run it again:

    python playground.py

Ideas to try are listed at the bottom of this file.
"""

import jax

jax.config.update("jax_enable_x64", True)  # double precision for conservation studies

import jax.numpy as jnp  # noqa: E402

import hamjax as hj  # noqa: E402

# --- Figure-eight three-body orbit ---------------------------------------------
state, masses, period = hj.figure_eight()
H = hj.NBody(masses, G=1.0, softening=0.0)  # only the energy is defined

traj = hj.simulate(H, state, dt=period / 1000, n_steps=100_000, save_every=100)
print("trajectory shape:", traj.q.shape)  # (1001, 3, 2) = saved times x bodies x (x, y)
print("max energy error:", hj.relative_energy_error(H, traj).max())  # ~2e-5
print("initial forces (from jax.grad):\n", H.force(state.q))


# --- Your own system: just write the energy ------------------------------------
class AnharmonicChain(hj.SeparableHamiltonian):
    k: float = 1.0

    def kinetic(self, p):
        return 0.5 * jnp.sum(p**2)

    def potential(self, q):
        return jnp.sum(0.5 * self.k * jnp.diff(q) ** 2 + 0.25 * jnp.diff(q) ** 4)


chain = AnharmonicChain()
q0, p0 = jnp.linspace(0, 1, 8) ** 2, jnp.zeros(8)
traj2 = hj.simulate(chain, hj.PhaseSpace(q0, p0), dt=0.01, n_steps=10_000)
print("chain trajectory shape:", traj2.q.shape)  # (10001, 8)
print("chain max energy error:", hj.relative_energy_error(chain, traj2).max())

# --- Ideas to try ----------------------------------------------------------------
# * Use a coarser step, dt=period / 50, and watch the energy error grow (Verlet is 2nd order).
# * Pass integrator=hj.RK4() or integrator=hj.Yoshida4() to hj.simulate to compare methods.
# * Replace hj.figure_eight() with hj.kepler_two_body(eccentricity=0.9).
# * Nudge a body, state = hj.PhaseSpace(state.q, state.p.at[0, 0].add(0.01)), and see the orbit break up.
# * Try AnharmonicChain(k=5.0), or a longer chain.
