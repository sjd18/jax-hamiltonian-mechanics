"""Time integrators and ``lax.scan`` rollouts.

Every integrator is a stateless :class:`equinox.Module` with three pure
methods::

    carry = integrator.init(H, state)        # e.g. cache the initial force
    carry = integrator.step(H, carry, dt)    # advance by one step
    state = integrator.state(carry)          # project back to phase space

Keeping the force in the carry means velocity Verlet costs exactly **one**
force evaluation per step: the closing half-kick's force is reused as the
next step's opening half-kick.

:func:`simulate` wires an integrator into a nested ``jax.lax.scan``, so a
million-step rollout compiles to a single XLA ``while`` loop rather than a
million unrolled Python iterations.
"""

from __future__ import annotations

import abc
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .hamiltonians import PhaseSpace, SeparableHamiltonian, hamiltonian_vector_field

__all__ = [
    "AbstractIntegrator",
    "VelocityVerlet",
    "Yoshida4",
    "RK4",
    "Trajectory",
    "simulate",
    "rollout",
]


class _VerletCarry(NamedTuple):
    state: PhaseSpace
    force: Array  # -∂V/∂q evaluated at state.q


def _verlet_substep(H: SeparableHamiltonian, carry: _VerletCarry, dt) -> _VerletCarry:
    """Kick–drift–kick. Symplectic, time-reversible, second order."""
    q, p = carry.state
    p_half = p + 0.5 * dt * carry.force  # half kick
    q_new = q + dt * H.velocity(p_half)  # full drift
    f_new = H.force(q_new)  # the single new force evaluation
    p_new = p_half + 0.5 * dt * f_new  # half kick
    return _VerletCarry(PhaseSpace(q_new, p_new), f_new)


class AbstractIntegrator(eqx.Module):
    @abc.abstractmethod
    def init(self, hamiltonian, state: PhaseSpace) -> Any: ...

    @abc.abstractmethod
    def step(self, hamiltonian, carry: Any, dt) -> Any: ...

    @abc.abstractmethod
    def state(self, carry: Any) -> PhaseSpace: ...


class VelocityVerlet(AbstractIntegrator):
    """Velocity Verlet (a.k.a. Störmer–Verlet, kick–drift–kick leapfrog).

    Requires a :class:`~hamjax.SeparableHamiltonian`. Local error ``O(dt³)``,
    global error ``O(dt²)``; being symplectic, it exactly conserves a *shadow*
    Hamiltonian ``H̃ = H + O(dt²)``, so the energy error stays bounded for
    exponentially long times instead of drifting.
    """

    def init(self, hamiltonian, state):
        return _VerletCarry(state, hamiltonian.force(state.q))

    def step(self, hamiltonian, carry, dt):
        return _verlet_substep(hamiltonian, carry, dt)

    def state(self, carry):
        return carry.state


# Yoshida (1990) triple-jump coefficients.
_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _CBRT2)
_W0 = -_CBRT2 / (2.0 - _CBRT2)


class Yoshida4(AbstractIntegrator):
    """Fourth-order symplectic integrator: three Verlet steps of sizes
    ``w₁dt, w₀dt, w₁dt`` with ``2w₁ + w₀ = 1``. Three forces per step."""

    def init(self, hamiltonian, state):
        return _VerletCarry(state, hamiltonian.force(state.q))

    def step(self, hamiltonian, carry, dt):
        carry = _verlet_substep(hamiltonian, carry, _W1 * dt)
        carry = _verlet_substep(hamiltonian, carry, _W0 * dt)
        return _verlet_substep(hamiltonian, carry, _W1 * dt)

    def state(self, carry):
        return carry.state


class RK4(AbstractIntegrator):
    """Classical Runge–Kutta 4 on ``ż = J∇H``.

    Works for *any* Hamiltonian (including non-separable learned ones such as
    :class:`~hamjax.HNN`) but is **not** symplectic: energy drifts secularly.
    Included as the non-geometric baseline.
    """

    def init(self, hamiltonian, state):
        return state

    def step(self, hamiltonian, z, dt):
        f = lambda s: hamiltonian_vector_field(hamiltonian, s)  # noqa: E731
        axpy = lambda a, x, y: jax.tree.map(lambda xi, yi: yi + a * xi, x, y)  # noqa: E731
        k1 = f(z)
        k2 = f(axpy(0.5 * dt, k1, z))
        k3 = f(axpy(0.5 * dt, k2, z))
        k4 = f(axpy(dt, k3, z))
        return jax.tree.map(lambda zi, a, b, c, d: zi + dt / 6.0 * (a + 2 * b + 2 * c + d), z, k1, k2, k3, k4)

    def state(self, carry):
        return carry


class Trajectory(NamedTuple):
    """A sampled trajectory. Every leaf has a leading time axis."""

    t: Array
    state: PhaseSpace

    @property
    def q(self) -> Array:
        return self.state.q

    @property
    def p(self) -> Array:
        return self.state.p


def rollout(step_fn, init, n_steps: int, save_every: int = 1):
    """Iterate ``carry ← step_fn(carry)`` ``n_steps`` times with ``lax.scan``.

    Returns the stacked carries ``[init, c_{s}, c_{2s}, …]`` where ``s =
    save_every``. An inner ``scan`` advances ``save_every`` steps without
    emitting anything, so memory is ``O(n_steps / save_every)``.
    """
    if n_steps % save_every:
        raise ValueError(f"n_steps={n_steps} must be divisible by save_every={save_every}")

    def advance(carry, _):
        return step_fn(carry), None

    def chunk(carry, _):
        carry, _ = jax.lax.scan(advance, carry, length=save_every)
        return carry, carry

    _, saved = jax.lax.scan(chunk, init, length=n_steps // save_every)
    return jax.tree.map(lambda x0, xs: jnp.concatenate([x0[None], xs]), init, saved)


@eqx.filter_jit
def simulate(
    hamiltonian,
    state0: PhaseSpace,
    *,
    dt,
    n_steps: int,
    integrator: AbstractIntegrator = VelocityVerlet(),
    save_every: int = 1,
    t0=0.0,
) -> Trajectory:
    """Integrate Hamilton's equations from ``state0``.

    Args:
        hamiltonian: any callable ``H(PhaseSpace) -> scalar``. Splitting
            integrators additionally need a :class:`SeparableHamiltonian`.
        state0: initial condition.
        dt: time step. Pass a JAX array (rather than a Python float) to make
            it traced — e.g. to ``vmap`` over step sizes without recompiling.
        n_steps: number of steps (static).
        integrator: an :class:`AbstractIntegrator` instance.
        save_every: record every ``save_every``-th state (static).
        t0: initial time.

    Returns:
        A :class:`Trajectory` with ``n_steps // save_every + 1`` samples,
        including the initial condition.
    """
    carry0 = integrator.init(hamiltonian, state0)
    carries = rollout(lambda c: integrator.step(hamiltonian, c, dt), carry0, n_steps, save_every)
    states = jax.vmap(integrator.state)(carries)
    t = t0 + dt * save_every * jnp.arange(n_steps // save_every + 1)
    return Trajectory(t=t, state=states)
