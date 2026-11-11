"""Initial conditions for classic N-body problems (units with G = 1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .hamiltonians import PhaseSpace

__all__ = ["figure_eight", "kepler_two_body", "plummer_sphere", "remove_center_of_mass"]


def remove_center_of_mass(state: PhaseSpace, masses: Array) -> PhaseSpace:
    """Shift to the barycentric frame (zero COM position and total momentum)."""
    m = masses[:, None]
    com = jnp.sum(m * state.q, axis=0) / jnp.sum(masses)
    p_tot = jnp.sum(state.p, axis=0)
    return PhaseSpace(state.q - com, state.p - m * p_tot / jnp.sum(masses))


def figure_eight() -> tuple[PhaseSpace, Array, float]:
    """The Chenciner–Montgomery (2000) figure-eight choreography.

    Three equal masses chase each other around a single figure-eight curve.

    Returns:
        ``(state, masses, period)``.
    """
    x1 = jnp.array([0.97000436, -0.24308753])
    v3 = jnp.array([-0.93240737, -0.86473146])
    q = jnp.stack([x1, -x1, jnp.zeros(2)])
    v = jnp.stack([-v3 / 2, -v3 / 2, v3])
    masses = jnp.ones(3)
    return PhaseSpace(q, masses[:, None] * v), masses, 6.32591398


def kepler_two_body(
    eccentricity: float = 0.5, m1: float = 1.0, m2: float = 1.0, a: float = 1.0
) -> tuple[PhaseSpace, Array, float]:
    """A bound two-body orbit started at periapsis, in the barycentric frame.

    Returns:
        ``(state, masses, period)``.
    """
    M = m1 + m2
    r_p = a * (1 - eccentricity)
    v_p = jnp.sqrt(M / a * (1 + eccentricity) / (1 - eccentricity))  # vis-viva
    mu = m1 * m2 / M
    q = jnp.array([[-m2 / M * r_p, 0.0], [m1 / M * r_p, 0.0]])
    p = jnp.array([[0.0, -mu * v_p], [0.0, mu * v_p]])
    masses = jnp.array([m1, m2])
    period = 2 * jnp.pi * jnp.sqrt(a**3 / M)
    return PhaseSpace(q, p), masses, float(period)


def _isotropic(key: PRNGKeyArray, radius: Array) -> Array:
    """Points uniformly distributed on spheres of the given radii."""
    k1, k2 = jax.random.split(key)
    cos_t = jax.random.uniform(k1, radius.shape, minval=-1.0, maxval=1.0)
    phi = jax.random.uniform(k2, radius.shape, maxval=2 * jnp.pi)
    sin_t = jnp.sqrt(1.0 - cos_t**2)
    return radius[:, None] * jnp.stack([sin_t * jnp.cos(phi), sin_t * jnp.sin(phi), cos_t], -1)


def plummer_sphere(key: PRNGKeyArray, n: int, total_mass: float = 1.0) -> tuple[PhaseSpace, Array]:
    """Sample an equal-mass Plummer sphere in Hénon units (G = M = 1, E = -1/4).

    Positions follow the cumulative mass profile; speeds are drawn with
    von Neumann rejection (Aarseth, Hénon & Wielen 1974). Rejection sampling
    is vectorised: each particle gets a batch of candidates and keeps the
    first accepted one, so the whole sampler is jittable.

    Returns:
        ``(state, masses)`` in the barycentric frame.
    """
    k_r, k_pos, k_cand, k_dir = jax.random.split(key, 4)
    scale = 3 * jnp.pi / 16  # Plummer radius in Hénon units

    u = jax.random.uniform(k_r, (n,), minval=1e-6, maxval=1.0 - 1e-6)
    r = 1.0 / jnp.sqrt(u ** (-2.0 / 3.0) - 1.0)
    q = _isotropic(k_pos, r)

    n_cand = 64  # acceptance ≈ 43%, so P(all rejected) ≈ 1e-16
    x, y = jax.random.uniform(k_cand, (2, n, n_cand))
    accept = 0.1 * y < x**2 * (1.0 - x**2) ** 3.5
    x = x[jnp.arange(n), jnp.argmax(accept, axis=1)]
    v_esc = jnp.sqrt(2.0) * (1.0 + r**2) ** -0.25
    v = _isotropic(k_dir, x * v_esc)

    # Rescale from Plummer units (a = 1) to Hénon units.
    q, v = q * scale, v / jnp.sqrt(scale)
    masses = jnp.full((n,), total_mass / n)
    return remove_center_of_mass(PhaseSpace(q, masses[:, None] * v), masses), masses
