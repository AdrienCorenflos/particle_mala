from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.linalg import solve
from jax.scipy.stats import norm

from gradient_csmc.utils.math import mvn_logpdf


@partial(jax.jit, static_argnums=(5, 6))
def get_data(key, nu, phi, tau, rho, dim, T):
    m0, P0, F, Q, b = get_dynamics(nu, phi, tau, rho, dim)

    init_key, sampling_key = jax.random.split(key)

    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Q = jnp.linalg.cholesky(Q)

    inv_chol_P0 = solve(chol_P0, jnp.eye(dim), assume_a="pos")
    inv_chol_Q = solve(chol_Q, jnp.eye(dim), assume_a="pos")

    x0 = m0 + chol_P0 @ jax.random.normal(init_key, (dim,))

    def body(x_k, key_k):
        state_key, observation_key = jax.random.split(key_k)

        observation_scale = jnp.exp(0.5 * x_k)

        y_k = observation_scale * jax.random.normal(observation_key, shape=(dim,))
        x_kp1 = F @ x_k + b + chol_Q @ jax.random.normal(state_key, shape=(dim,))
        return x_kp1, (x_k, y_k)

    _, (xs, ys) = jax.lax.scan(body, x0, jax.random.split(sampling_key, T))
    return xs, ys, inv_chol_P0, inv_chol_Q


@partial(jax.jit, static_argnums=(4,))
def get_dynamics(nu, phi, tau, rho, dim):
    F = phi * jnp.eye(dim)
    Q, P0 = stationary_covariance(phi, tau, rho, dim)
    mu = nu * jnp.ones((dim,))
    b = mu + F @ mu
    return mu, P0, F, Q, b


@partial(jax.jit, static_argnums=(3,))
def stationary_covariance(phi, tau, rho, dim):
    U = tau * rho * jnp.ones((dim, dim))
    U = U.at[np.diag_indices(dim)].set(tau)
    vec_U = jnp.reshape(U, (dim ** 2, 1))
    vec_U_star = vec_U / (1 - phi ** 2)
    U_star = jnp.reshape(vec_U_star, (dim, dim))
    return U, U_star


@partial(jnp.vectorize, signature="(n),(n)->()")
def log_potential(x, y):
    scale = jnp.exp(0.5 * x)
    val = norm.logpdf(y, scale=scale)
    return jnp.nansum(val)  # in case the scale is infinite, we get nan, but we want 0


def log_likelihood(x, y):
    return jnp.sum(log_potential(x, y))


def log_pdf(xs, ys, m0, inv_chol_P0, F, inv_chol_Q, b):
    def _logpdf(zs):
        out = mvn_logpdf(zs[0], m0, None, inv_chol_P0, constant=False)
        pred_xs = jnp.einsum('...j,ij,i->...i', zs[:-1], F, b)
        out += jnp.sum(mvn_logpdf(zs[1:], pred_xs, None, inv_chol_Q, constant=False))
        out += log_likelihood(zs, ys)
        return out

    return jnp.vectorize(_logpdf, signature="(T,d)->()")(xs)
