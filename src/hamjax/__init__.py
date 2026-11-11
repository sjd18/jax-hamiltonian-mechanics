"""hamjax — Hamiltonian mechanics in JAX.

Write down an energy; get forces from ``jax.grad``, trajectories from
``jax.lax.scan``, ensembles from ``jax.vmap`` and speed from ``jax.jit``.
"""

from .diagnostics import (
    angular_momentum,
    center_of_mass,
    energy,
    relative_energy_error,
    total_momentum,
)
from .hamiltonians import (
    HarmonicOscillator,
    NBody,
    Pendulum,
    PhaseSpace,
    SeparableHamiltonian,
    hamiltonian_vector_field,
)
from .hnn import HNN, BaselineMLP, SeparableHNN, hnn_loss
from .initial_conditions import figure_eight, kepler_two_body, plummer_sphere, remove_center_of_mass
from .integrators import RK4, AbstractIntegrator, Trajectory, VelocityVerlet, Yoshida4, rollout, simulate

__version__ = "0.1.0"

__all__ = [
    "PhaseSpace",
    "SeparableHamiltonian",
    "NBody",
    "HarmonicOscillator",
    "Pendulum",
    "hamiltonian_vector_field",
    "AbstractIntegrator",
    "VelocityVerlet",
    "Yoshida4",
    "RK4",
    "Trajectory",
    "simulate",
    "rollout",
    "energy",
    "relative_energy_error",
    "total_momentum",
    "angular_momentum",
    "center_of_mass",
    "figure_eight",
    "kepler_two_body",
    "plummer_sphere",
    "remove_center_of_mass",
    "HNN",
    "SeparableHNN",
    "BaselineMLP",
    "hnn_loss",
]
