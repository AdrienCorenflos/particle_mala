from functools import partial

import jax
from jax import numpy as jnp
from jax.scipy.stats import norm


@partial(jax.jit, static_argnums=(2, 3))
def get_data(key, sigma, dim, T):
    init_key, sampling_key = jax.random.split(key)

    x0 = sigma * jax.random.normal(init_key, (dim,))
    eps_xs, eps_ys = jax.random.normal(init_key, (2, T, dim,))

    def body(x_k, inps):
        eps_x, eps_y = inps
        y_k = x_k + eps_y
        x_kp1 = x_k + sigma * eps_x
        return x_kp1, (x_k, y_k)

    _, (xs, ys) = jax.lax.scan(body, x0, (eps_xs, eps_ys))
    return xs, ys


@partial(jnp.vectorize, signature="(n),(n)->()")
def log_potential(x, y):
    val = norm.logpdf(y, x)
    return jnp.sum(val)


def log_likelihood(x, y):
    return jnp.sum(log_potential(x, y))


def log_pdf(xs, ys, sigma):
    def _logpdf(zs):
        out = jnp.sum(norm.logpdf(zs[0], scale=sigma))
        out += jnp.sum(norm.logpdf(zs[1:], zs[:-1], sigma))
        out += jnp.sum(norm.logpdf(zs, ys))
        return out

    return jnp.vectorize(_logpdf, signature="(T,d)->()")(xs)
