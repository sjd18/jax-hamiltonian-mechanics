import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

import hamjax as hj


def newtonian_forces(q, m, G=1.0, eps=0.0):
    """Hand-written pairwise forces, the thing autodiff saves us from."""
    n = q.shape[0]
    F = np.zeros_like(q)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = q[j] - q[i]
                r2 = d @ d + eps**2
                F[i] += G * m[i] * m[j] * d / r2**1.5
    return F


@pytest.mark.parametrize("eps", [0.0, 0.05])
@pytest.mark.parametrize("dim", [2, 3])
def test_autodiff_force_matches_newton(eps, dim):
    key = jax.random.PRNGKey(0)
    kq, km = jax.random.split(key)
    q = jax.random.normal(kq, (7, dim))
    m = jax.random.uniform(km, (7,), minval=0.5, maxval=2.0)
    H = hj.NBody(m, G=1.3, softening=eps)
    np.testing.assert_allclose(
        H.force(q), newtonian_forces(np.asarray(q), np.asarray(m), 1.3, eps), rtol=1e-10, atol=1e-12
    )


def test_force_is_finite_without_softening():
    """The double-where trick keeps the r=0 self-pair out of the gradient."""
    s, m, _ = hj.figure_eight()
    F = hj.NBody(m).force(s.q)
    assert jnp.all(jnp.isfinite(F))


def test_velocity_is_p_over_m():
    m = jnp.array([1.0, 2.0, 4.0])
    p = jnp.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose(hj.NBody(m).velocity(p), p / m[:, None])


def test_vector_field_is_divergence_free():
    """Liouville: Hamiltonian flow preserves phase-space volume, tr(∂ż/∂z) = 0."""
    s, m, _ = hj.figure_eight()
    H = hj.NBody(m)
    flat, unravel = jax.flatten_util.ravel_pytree(s)
    f = lambda z: jax.flatten_util.ravel_pytree(hj.hamiltonian_vector_field(H, unravel(z)))[0]  # noqa: E731
    assert abs(jnp.trace(jax.jacfwd(f)(flat))) < 1e-10


def test_energy_is_constant_along_vector_field():
    """dH/dt = ∇H · J∇H = 0 identically."""
    s, m, _ = hj.figure_eight()
    H = hj.NBody(m)
    _, jvp = jax.jvp(H, (s,), (hj.hamiltonian_vector_field(H, s),))
    assert abs(jvp) < 1e-12
