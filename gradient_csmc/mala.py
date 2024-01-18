"""
Implements the multi-try MALA kernel with or without marginalised acceptance.
"""
from functools import partial

import jax
import jax.numpy as jnp
from chex import PRNGKey

from gradient_csmc.tp import init, delta_adaptation_routine, sampling_routine
from gradient_csmc.utils.common import force_move

_ = init, delta_adaptation_routine, sampling_routine


def kernel(key: PRNGKey, state, delta: float, log_pdf, N: int = 1, auxiliary: bool = False):
    """
    Implements the multi-try MALA kernel with or without marginalised acceptance.

    Parameters
    ----------
    key: PRNGKey
        JAX PRNGKey.
    state:
        Reference trajectory.
    delta:
        Gradient step-size.
    log_pdf:
        Log-density function.
    N:
        Number of trys.
    auxiliary:
        Whether to use the auxiliary marginalised acceptance.

    Returns
    -------
    x_star: Array
        New reference trajectory.
    accept: bool
        Whether the new trajectory was accepted.
    """
    ###############################
    #        HOUSEKEEPING         #
    ###############################
    val_and_grad_log_pdf = jnp.vectorize(jax.value_and_grad(log_pdf), signature='(T,d)->(),(T,d)')

    x_star, val_x_star, grad_x_star = state
    T, d_x = x_star.shape

    key_sample, key_choice, key_aux = jax.random.split(key, 3)
    std_dev = jnp.sqrt(0.5 * delta)

    ###############################
    #        Auxiliary step       #
    ###############################
    eps_aux = jax.random.normal(key_aux, shape=(T, d_x))
    us = x_star + 0.5 * delta * grad_x_star + std_dev * eps_aux

    samples = us + std_dev * jax.random.normal(key_sample, shape=(N, T, d_x))
    vals, grads = val_and_grad_log_pdf(samples)

    samples = jnp.insert(samples, 0, x_star, axis=0)
    vals = jnp.insert(vals, 0, val_x_star, axis=0)
    grads = jnp.insert(grads, 0, grad_x_star, axis=0)

    if auxiliary:
        correction_term = -0.5 * jnp.sum((us - samples - 0.5 * delta * grads) ** 2, (1, 2)) / std_dev ** 2
        log_weights = vals + correction_term
        proposal_log_pdf = -0.5 * jnp.sum((us - samples) ** 2, (1, 2)) / std_dev ** 2
        log_weights -= proposal_log_pdf
    else:
        samples = jnp.reshape(samples, (N + 1, T * d_x))
        grads = jnp.reshape(grads, (N + 1, T * d_x))

        log_H = get_log_H(samples, 2 / delta, 0.5 * delta)
        log_weights = vals + log_H(samples, 0.5 * delta * grads)

        samples = jnp.reshape(samples, (N + 1, T, d_x))
        grads = jnp.reshape(grads, (N + 1, T, d_x))

    weights = jnp.exp(log_weights - jnp.max(log_weights))
    weights /= jnp.sum(weights)

    idx, alpha = force_move(key_choice, weights, 0)
    x_star = samples[idx]
    val_x_star = vals[idx]
    grad_x_star = grads[idx]

    return x_star, val_x_star, grad_x_star, idx != 0


def get_log_H(xs, c_inv, e):
    N, D = xs.shape
    N = N - 1
    x_bar = jnp.mean(xs, 0)
    G = 1 / (e + N * e)

    @partial(jnp.vectorize, signature='(n),(n)->()')
    def log_H(x, phi):
        B_x_phi = x + phi

        term_1 = -(c_inv + G) * jnp.dot(x, x)
        term_2 = N * (c_inv - N * G) * jnp.dot(B_x_phi, B_x_phi)
        term_3 = 2 * (N + 1) * jnp.dot(x_bar, G * x - (c_inv - N * G) * B_x_phi)
        term_4 = 2 * (c_inv - N * G) * jnp.dot(x, B_x_phi)

        return -0.5 * jnp.nan_to_num(term_1 + term_2 + term_3 + term_4)

    return log_H
