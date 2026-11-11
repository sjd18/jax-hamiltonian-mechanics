import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

import hamjax as hj

SYMPLECTIC = [hj.VelocityVerlet(), hj.Yoshida4()]


def one_step_map(H, integrator, dt, unravel):
    def f(z):
        carry = integrator.step(H, integrator.init(H, unravel(z)), dt)
        return jax.flatten_util.ravel_pytree(integrator.state(carry))[0]

    return f


@pytest.mark.parametrize("integrator", SYMPLECTIC, ids=lambda i: type(i).__name__)
def test_step_is_symplectic(integrator):
    """The Jacobian M of the one-step map satisfies Mᵀ Ω M = Ω."""
    s, m, _ = hj.figure_eight()
    H = hj.NBody(m)
    z, unravel = jax.flatten_util.ravel_pytree(s)
    M = jax.jacfwd(one_step_map(H, integrator, 0.05, unravel))(z)
    n = z.size // 2
    Omega = jnp.block([[jnp.zeros((n, n)), jnp.eye(n)], [-jnp.eye(n), jnp.zeros((n, n))]])
    np.testing.assert_allclose(M.T @ Omega @ M, Omega, atol=1e-11)


def test_rk4_is_not_symplectic():
    s, m, _ = hj.figure_eight()
    H = hj.NBody(m)
    z, unravel = jax.flatten_util.ravel_pytree(s)
    M = jax.jacfwd(one_step_map(H, hj.RK4(), 0.05, unravel))(z)
    n = z.size // 2
    Omega = jnp.block([[jnp.zeros((n, n)), jnp.eye(n)], [-jnp.eye(n), jnp.zeros((n, n))]])
    assert jnp.abs(M.T @ Omega @ M - Omega).max() > 1e-8


@pytest.mark.parametrize("integrator", SYMPLECTIC, ids=lambda i: type(i).__name__)
def test_time_reversible(integrator):
    s, m, _ = hj.figure_eight()
    H = hj.NBody(m)
    fwd = hj.simulate(H, s, dt=0.01, n_steps=500, integrator=integrator)
    end = hj.PhaseSpace(fwd.q[-1], fwd.p[-1])
    back = hj.simulate(H, end, dt=-0.01, n_steps=500, integrator=integrator)
    np.testing.assert_allclose(back.q[-1], s.q, atol=1e-10)
    np.testing.assert_allclose(back.p[-1], s.p, atol=1e-10)


def test_verlet_conserves_momenta_exactly():
    """Pairwise central forces ⇒ Verlet conserves P and L to round-off."""
    s, m = hj.plummer_sphere(jax.random.PRNGKey(1), 64)
    H = hj.NBody(m, softening=0.01)
    tr = hj.simulate(H, s, dt=1e-3, n_steps=2000, save_every=100)
    P = hj.total_momentum(tr.state)
    L = hj.angular_momentum(tr.state)
    np.testing.assert_allclose(P, jnp.broadcast_to(P[0], P.shape), atol=1e-13)
    np.testing.assert_allclose(L, jnp.broadcast_to(L[0], L.shape), atol=1e-13)


def test_verlet_energy_error_bounded_rk4_drifts():
    """Over many orbits Verlet's energy error oscillates; RK4's grows."""
    s, m, T = hj.kepler_two_body(eccentricity=0.6)
    H = hj.NBody(m)
    dt, n = T / 200, 200 * 200  # 200 orbits
    ev = hj.relative_energy_error(H, hj.simulate(H, s, dt=dt, n_steps=n, save_every=200))
    er = hj.relative_energy_error(H, hj.simulate(H, s, dt=dt, n_steps=n, save_every=200, integrator=hj.RK4()))
    # Verlet: the error after 200 orbits is no worse than after 10.
    assert ev[-1] < 2 * ev[:11].max() + 1e-12
    # RK4: monotone secular drift.
    assert er[-1] > 10 * er[10]


@pytest.mark.parametrize(
    "integrator, order",
    [(hj.VelocityVerlet(), 2), (hj.Yoshida4(), 4), (hj.RK4(), 4)],
    ids=["vv", "y4", "rk4"],
)
def test_convergence_order(integrator, order):
    """vmap over dt: the global error at t = 2π scales as dt^order."""
    H = hj.HarmonicOscillator()
    s = hj.PhaseSpace(jnp.array([1.0]), jnp.array([0.0]))
    ns = [64, 128, 256]
    errs = []
    for n in ns:
        tr = hj.simulate(H, s, dt=2 * jnp.pi / n, n_steps=n, integrator=integrator, save_every=n)
        errs.append(float(jnp.hypot(tr.q[-1, 0] - 1.0, tr.p[-1, 0])))
    slopes = -np.diff(np.log(errs)) / np.log(2)
    np.testing.assert_allclose(slopes, order, atol=0.15)


def test_save_every_consistent():
    s, m, _ = hj.figure_eight()
    H = hj.NBody(m)
    a = hj.simulate(H, s, dt=0.01, n_steps=100, save_every=1)
    b = hj.simulate(H, s, dt=0.01, n_steps=100, save_every=10)
    assert b.q.shape[0] == 11
    np.testing.assert_allclose(a.q[::10], b.q, atol=1e-14)
    np.testing.assert_allclose(a.t[::10], b.t)


def test_save_every_must_divide():
    s, m, _ = hj.figure_eight()
    with pytest.raises(ValueError):
        hj.simulate(hj.NBody(m), s, dt=0.01, n_steps=100, save_every=7)


def test_vmap_over_initial_conditions_and_dt():
    s, m, _ = hj.figure_eight()
    H = hj.NBody(m)
    batch = jax.tree.map(lambda x: jnp.stack([x, 1.01 * x, 0.99 * x]), s)
    dts = jnp.array([0.01, 0.005, 0.002])
    run = jax.vmap(lambda st, dt: hj.simulate(H, st, dt=dt, n_steps=50).q)
    out = run(batch, dts)
    assert out.shape == (3, 51, 3, 2)
    np.testing.assert_allclose(out[0], hj.simulate(H, s, dt=0.01, n_steps=50).q, atol=1e-14)


def test_grad_through_rollout():
    """Differentiate a final-time loss w.r.t. the initial momenta through lax.scan."""
    s, m, _ = hj.figure_eight()
    H = hj.NBody(m)

    def loss(p0):
        tr = hj.simulate(H, hj.PhaseSpace(s.q, p0), dt=0.01, n_steps=100, save_every=100)
        return jnp.sum(tr.q[-1] ** 2)

    g = jax.grad(loss)(s.p)
    fd_dir = jax.random.normal(jax.random.PRNGKey(0), s.p.shape)
    h = 1e-6
    fd = (loss(s.p + h * fd_dir) - loss(s.p - h * fd_dir)) / (2 * h)
    np.testing.assert_allclose(jnp.vdot(g, fd_dir), fd, rtol=1e-6)


def test_figure_eight_returns_after_one_period():
    s, m, T = hj.figure_eight()
    n = 4000
    tr = hj.simulate(hj.NBody(m), s, dt=T / n, n_steps=n, integrator=hj.Yoshida4(), save_every=n)
    np.testing.assert_allclose(tr.q[-1], s.q, atol=1e-6)
