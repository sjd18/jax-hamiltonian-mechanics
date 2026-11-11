import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import hamjax as hj


def test_hnn_is_a_hamiltonian():
    """An HNN plugs into simulate() and RK4 like any other Hamiltonian."""
    model = hj.HNN(dim=1, width=16, key=jax.random.PRNGKey(0))
    s = hj.PhaseSpace(jnp.array([0.5]), jnp.array([0.1]))
    tr = hj.simulate(model, s, dt=0.01, n_steps=100, integrator=hj.RK4())
    assert tr.q.shape == (101, 1)


def test_hnn_vector_field_conserves_learned_energy():
    """Even untrained, ż = J∇H_θ is orthogonal to ∇H_θ."""
    model = hj.HNN(dim=2, width=32, key=jax.random.PRNGKey(0))
    s = hj.PhaseSpace(jnp.array([0.3, -0.2]), jnp.array([0.1, 0.4]))
    _, dHdt = jax.jvp(model, (s,), (model.time_derivative(s),))
    assert abs(dHdt) < 1e-12


def test_separable_hnn_runs_under_verlet():
    model = hj.SeparableHNN(dim=1, width=16, key=jax.random.PRNGKey(0))
    s = hj.PhaseSpace(jnp.array([0.5]), jnp.array([0.1]))
    tr = hj.simulate(model, s, dt=0.01, n_steps=100, integrator=hj.VelocityVerlet())
    assert jnp.all(jnp.isfinite(tr.q))


def test_hnn_learns_harmonic_oscillator():
    key = jax.random.PRNGKey(0)
    kd, km = jax.random.split(key)
    H_true = hj.HarmonicOscillator()
    states = hj.PhaseSpace(*jax.random.uniform(kd, (2, 256, 1), minval=-1, maxval=1))
    targets = jax.vmap(lambda s: hj.hamiltonian_vector_field(H_true, s))(states)

    model = hj.HNN(dim=1, width=64, key=km)
    opt = optax.adam(3e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state):
        loss, g = eqx.filter_value_and_grad(hj.hnn_loss)(model, states, targets)
        updates, opt_state = opt.update(g, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss

    for _ in range(600):
        model, opt_state, loss = step(model, opt_state)
    assert loss < 1e-3
    # The learned H is H_true up to an additive constant.
    h = jax.vmap(model)(states)
    h0 = jax.vmap(H_true)(states)
    np.testing.assert_allclose(h - h.mean(), h0 - h0.mean(), atol=0.03)
