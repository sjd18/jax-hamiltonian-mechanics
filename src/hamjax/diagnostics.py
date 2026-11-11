"""Conserved quantities and error measures."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .hamiltonians import PhaseSpace
from .integrators import Trajectory

__all__ = [
    "energy",
    "relative_energy_error",
    "total_momentum",
    "angular_momentum",
    "center_of_mass",
]


def energy(hamiltonian, traj: Trajectory) -> Array:
    """``H`` evaluated at every saved state (vmapped over time)."""
    return jax.vmap(hamiltonian)(traj.state)


def relative_energy_error(hamiltonian, traj: Trajectory) -> Array:
    """``|H(t) - H(0)| / |H(0)|``."""
    e = energy(hamiltonian, traj)
    return jnp.abs(e - e[0]) / jnp.abs(e[0])


def total_momentum(state: PhaseSpace) -> Array:
    """``Σᵢ pᵢ`` over bodies (works batched over leading axes)."""
    return jnp.sum(state.p, axis=-2)


def angular_momentum(state: PhaseSpace) -> Array:
    """``Σᵢ qᵢ × pᵢ``. Scalar in 2-D, a 3-vector in 3-D."""
    q, p = state
    if q.shape[-1] == 2:
        return jnp.sum(q[..., 0] * p[..., 1] - q[..., 1] * p[..., 0], axis=-1)
    return jnp.sum(jnp.cross(q, p), axis=-2)


def center_of_mass(state: PhaseSpace, masses: Array) -> Array:
    m = masses[:, None]
    return jnp.sum(m * state.q, axis=-2) / jnp.sum(masses)
