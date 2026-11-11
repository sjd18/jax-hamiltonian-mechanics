"""Hamiltonians as Equinox modules.

The convention throughout ``hamjax`` is that *a Hamiltonian is any callable*
``H(state: PhaseSpace) -> scalar``. Everything dynamical — forces, velocities,
the full vector field — is obtained from that scalar by automatic
differentiation, never written by hand:

    dq/dt =  ∂H/∂p ,      dp/dt = -∂H/∂q .

Separable Hamiltonians ``H(q, p) = T(p) + V(q)`` additionally expose
``kinetic`` and ``potential`` so that splitting integrators (velocity Verlet,
Yoshida) can kick and drift independently.
"""

from __future__ import annotations

import abc
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    "PhaseSpace",
    "SeparableHamiltonian",
    "NBody",
    "HarmonicOscillator",
    "Pendulum",
    "hamiltonian_vector_field",
]


class PhaseSpace(NamedTuple):
    """A point ``(q, p)`` in phase space.

    A ``NamedTuple`` is a JAX pytree, so ``PhaseSpace`` flows through ``jit``,
    ``grad``, ``vmap`` and ``lax.scan`` with no registration boilerplate. For an
    N-body system ``q`` and ``p`` have shape ``(N, dim)``.
    """

    q: Array
    p: Array


def hamiltonian_vector_field(hamiltonian, state: PhaseSpace) -> PhaseSpace:
    """Hamilton's equations, ``ż = J ∇H(z)``, via a single reverse-mode pass.

    ``jax.grad`` of a function of a ``PhaseSpace`` returns a ``PhaseSpace`` of
    partial derivatives, so the symplectic matrix ``J`` reduces to a swap and a
    sign flip.
    """
    dH = jax.grad(hamiltonian)(state)
    return PhaseSpace(q=dH.p, p=-dH.q)


class SeparableHamiltonian(eqx.Module):
    """Base class for ``H(q, p) = T(p) + V(q)``.

    Subclasses implement :meth:`kinetic` and :meth:`potential`; the force and
    velocity used by the integrators are derived with ``jax.grad``.
    """

    @abc.abstractmethod
    def kinetic(self, p: Array) -> Float[Array, ""]:
        """Kinetic energy ``T(p)``."""

    @abc.abstractmethod
    def potential(self, q: Array) -> Float[Array, ""]:
        """Potential energy ``V(q)``."""

    def __call__(self, state: PhaseSpace) -> Float[Array, ""]:
        return self.kinetic(state.p) + self.potential(state.q)

    def force(self, q: Array) -> Array:
        """Generalised force ``F = -∂V/∂q`` — computed by autodiff."""
        return -jax.grad(self.potential)(q)

    def velocity(self, p: Array) -> Array:
        """Generalised velocity ``q̇ = ∂T/∂p`` — computed by autodiff."""
        return jax.grad(self.kinetic)(p)


class NBody(SeparableHamiltonian):
    r"""Newtonian gravity with Plummer softening.

    .. math::

        H = \sum_i \frac{|p_i|^2}{2 m_i}
            - \sum_{i<j} \frac{G m_i m_j}{\sqrt{|q_i - q_j|^2 + \varepsilon^2}}

    Only the *energy* is written down. The pairwise forces — an ``O(N²)``
    tangle of unit vectors if derived by hand — come out of ``jax.grad``.

    Attributes:
        masses: shape ``(N,)``.
        G: gravitational constant.
        softening: Plummer softening length ``ε`` (``0`` for exact Newtonian).
    """

    masses: Float[Array, " N"]
    G: float = 1.0
    softening: float = 0.0

    def __init__(self, masses, G: float = 1.0, softening: float = 0.0):
        self.masses = jnp.asarray(masses)
        self.G = G
        self.softening = softening

    def kinetic(self, p):
        return 0.5 * jnp.sum(p**2 / self.masses[:, None])

    def potential(self, q):
        n = q.shape[0]
        diff = q[:, None, :] - q[None, :, :]
        r2 = jnp.sum(diff**2, axis=-1) + self.softening**2
        # "Double-where" trick: the diagonal has r = 0, and although we mask
        # it out of the forward pass, d/dr (1/r) at r = 0 is inf and inf * 0
        # = nan would leak into the gradient. Substituting a safe value
        # *before* the singular op keeps both passes finite.
        self_pair = jnp.eye(n, dtype=bool)
        r2_safe = jnp.where(self_pair, 1.0, r2)
        inv_r = jnp.where(self_pair, 0.0, jax.lax.rsqrt(r2_safe))
        mm = self.masses[:, None] * self.masses[None, :]
        # The double sum counts each pair twice.
        return -0.5 * self.G * jnp.sum(mm * inv_r)


class HarmonicOscillator(SeparableHamiltonian):
    r"""``H = p²/2m + k q²/2`` — the "ideal mass-spring" task of Greydanus et al."""

    mass: float = 1.0
    k: float = 1.0

    def kinetic(self, p):
        return 0.5 * jnp.sum(p**2) / self.mass

    def potential(self, q):
        return 0.5 * self.k * jnp.sum(q**2)


class Pendulum(SeparableHamiltonian):
    r"""``H = p²/2ml² + mgl(1 - cos q)`` — the "ideal pendulum" task."""

    mass: float = 1.0
    length: float = 1.0
    g: float = 3.0

    def kinetic(self, p):
        return 0.5 * jnp.sum(p**2) / (self.mass * self.length**2)

    def potential(self, q):
        return self.mass * self.g * self.length * jnp.sum(1.0 - jnp.cos(q))
