"""Hamiltonian Neural Networks (Greydanus, Dzamba & Yosinski, 2019).

An HNN parameterises the *scalar* ``H_θ(q, p)`` with a neural network and
obtains the dynamics as its symplectic gradient,

    ż = J ∇_z H_θ(z),

trained by regressing ``ż`` onto observed time derivatives. Because the
model *is* a Hamiltonian under ``hamjax``'s conventions (a callable
``PhaseSpace -> scalar``), it drops straight into :func:`hamjax.simulate`.

:class:`SeparableHNN` constrains ``H_θ = T_θ(p) + V_θ(q)``, which buys the
learned model access to the symplectic integrators (velocity Verlet, Yoshida)
— the learned force is ``-∇V_θ``, again by autodiff.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .hamiltonians import PhaseSpace, SeparableHamiltonian, hamiltonian_vector_field

__all__ = ["HNN", "SeparableHNN", "BaselineMLP", "hnn_loss", "flatten_state"]


def flatten_state(state: PhaseSpace) -> Array:
    return jnp.concatenate([state.q.ravel(), state.p.ravel()])


class HNN(eqx.Module):
    """``H_θ : R^{2d} → R`` — a general (non-separable) learned Hamiltonian."""

    mlp: eqx.nn.MLP

    def __init__(self, dim: int, width: int = 200, depth: int = 2, *, key: PRNGKeyArray):
        self.mlp = eqx.nn.MLP(2 * dim, "scalar", width, depth, activation=jnp.tanh, key=key)

    def __call__(self, state: PhaseSpace) -> Array:
        return self.mlp(flatten_state(state))

    def time_derivative(self, state: PhaseSpace) -> PhaseSpace:
        return hamiltonian_vector_field(self, state)


class SeparableHNN(SeparableHamiltonian):
    """``H_θ = T_θ(p) + V_θ(q)`` with two independent scalar MLPs."""

    kinetic_net: eqx.nn.MLP
    potential_net: eqx.nn.MLP

    def __init__(self, dim: int, width: int = 200, depth: int = 2, *, key: PRNGKeyArray):
        kt, kv = jax.random.split(key)
        self.kinetic_net = eqx.nn.MLP(dim, "scalar", width, depth, activation=jnp.tanh, key=kt)
        self.potential_net = eqx.nn.MLP(dim, "scalar", width, depth, activation=jnp.tanh, key=kv)

    def kinetic(self, p):
        return self.kinetic_net(p.ravel())

    def potential(self, q):
        return self.potential_net(q.ravel())

    def time_derivative(self, state: PhaseSpace) -> PhaseSpace:
        return hamiltonian_vector_field(self, state)


class BaselineMLP(eqx.Module):
    """The paper's baseline: an MLP that predicts ``ż`` directly, with no
    Hamiltonian structure. It is *not* a Hamiltonian, so it is integrated
    with a plain RK4 on its own output (see ``examples/train_hnn.py``)."""

    mlp: eqx.nn.MLP
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, width: int = 200, depth: int = 2, *, key: PRNGKeyArray):
        self.dim = dim
        self.mlp = eqx.nn.MLP(2 * dim, 2 * dim, width, depth, activation=jnp.tanh, key=key)

    def time_derivative(self, state: PhaseSpace) -> PhaseSpace:
        out = self.mlp(flatten_state(state))
        return PhaseSpace(out[: self.dim].reshape(state.q.shape), out[self.dim :].reshape(state.p.shape))


def hnn_loss(model, states: PhaseSpace, targets: PhaseSpace) -> Array:
    """Mean squared error between predicted and observed ``(q̇, ṗ)``.

    ``states`` and ``targets`` are batched along a leading axis. ``model`` is
    any of :class:`HNN`, :class:`SeparableHNN`, :class:`BaselineMLP`.
    """
    pred = jax.vmap(model.time_derivative)(states)
    sq = jax.tree.map(lambda a, b: jnp.mean((a - b) ** 2), pred, targets)
    return sum(jax.tree.leaves(sq))
