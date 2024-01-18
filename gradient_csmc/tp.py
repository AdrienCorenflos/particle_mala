"""
Implements the filtering auxiliary Titsias-Papaspiliopoulos kernel with the Kalman formulation of Corenflos et al. (2023) and multi-try proposals.
"""

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey

from gradient_csmc.utils.common import force_move
from gradient_csmc.utils.kalman import filtering, sampling
from gradient_csmc.utils.pbar import progress_bar_scan


def get_kernel(m0: Array, P0: Array,
               Fs, Qs, bs, log_pdf,
               N: int = 1,
               ):
    """
    Get a multi-try ATP kernel.

    Parameters
    ----------
    m0:
        Initial distribution mean vector.
    P0:
        Initial distribution covariance matrix.
    Fs:
        Array of transition matrices.
    Qs:
        Array of covariance matrices. The first element is the covariance at time 0: p(x0) = N(m0, Qs[0]).
    bs:
        Array of offset vectors.
    log_pdf:
        Log-density function.
    N:
        Number of trys.

    Returns
    -------
    kernel: Callable
        ATP kernel function.
    """
    T, d_x, _ = Fs.shape
    T += 1

    Hs = jnp.repeat(jnp.eye(d_x)[None, ...], T, axis=0)
    eyes = jnp.repeat(jnp.eye(d_x)[None, ...], T, axis=0)
    cs = jnp.zeros((T, d_x))

    val_and_grad_log_pdf = jnp.vectorize(jax.value_and_grad(log_pdf), signature='(T,d)->(),(T,d)')

    def kernel(key: PRNGKey, state, delta: float):
        """
        Implements the multi-try ATP kernel.

        Parameters
        ----------
        key: PRNGKey
            JAX PRNGKey.
        state:
            Reference trajectory.
        delta:
            Gradient step-size.

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

        x_star, val_x_star, grad_x_star = state
        key_sample, key_choice, key_aux = jax.random.split(key, 3)
        std_dev = jnp.sqrt(0.5 * delta)

        eps_aux = jax.random.normal(key_aux, shape=(T, d_x))
        us = x_star + 0.5 * delta * grad_x_star + std_dev * eps_aux

        Rs = 0.5 * delta * eyes
        fms, fPs, _ = filtering.filtering(us, m0, P0, Fs, Qs, bs, Hs, Rs, cs)

        samples = sampling.sampling(key_sample, fms, fPs, Fs, Qs, bs, N)
        vals, grads = val_and_grad_log_pdf(samples)

        samples = jnp.insert(samples, 0, x_star, axis=0)
        vals = jnp.insert(vals, 0, val_x_star, axis=0)
        grads = jnp.insert(grads, 0, grad_x_star, axis=0)

        temp = (us[None, :] - samples - 0.25 * delta * grads)
        correction_term = jnp.einsum('...ij,...ij->...', temp, grads)

        log_weights = vals + correction_term
        weights = jnp.exp(log_weights - jnp.max(log_weights))
        weights /= jnp.sum(weights)

        idx, alpha = force_move(key_choice, weights, 0)
        x_star = samples[idx]
        val_x_star = vals[idx]
        grad_x_star = grads[idx]

        return x_star, val_x_star, grad_x_star, idx != 0

    return kernel


def init(x: Array, log_pdf):
    """
    Initialize the reference trajectory.

    Parameters
    ----------
    x:
        Initial trajectory.
    log_pdf:
        Log-density function.

    Returns
    -------
    state:
        Initial state.
    """
    val_x, grad_x = jax.value_and_grad(log_pdf)(x)
    return x, val_x, grad_x


def delta_adaptation_routine(
        key,
        init_xs,
        kernel,
        target_acceptance,
        initial_delta,
        n_steps,
        verbose=False,
        min_delta=1e-6,
        max_delta=1e3,
        window_size=100,
        rate=0.1,
        min_rate=1e-2,
        **_kwargs
):
    if verbose:
        decorator = progress_bar_scan(n_steps, show=2)
    else:
        decorator = lambda x: x

    @decorator
    def body(carry, inp):
        xs, delta, acceptance_rate, accepted_history = carry
        i, key_i = inp

        # Run kernel
        *next_xs, accepted = kernel(key_i, xs, delta)

        # acceptance_rate = (i * acceptance_rate + accepted) / (i + 1)

        accepted_history = accepted_history.at[1:].set(accepted_history[:-1])
        accepted_history = accepted_history.at[0].set(accepted)
        acceptance_rate = jnp.nanmean(accepted_history, 0)

        flag = jnp.logical_or(acceptance_rate < target_acceptance - 0.05,
                              acceptance_rate > target_acceptance + 0.05)
        flag &= i > window_size

        rate_i = rate / (i + 1) ** 0.5
        rate_i = jnp.maximum(min_rate, rate_i)
        deltas_otherwise = delta + rate_i * delta * (acceptance_rate - target_acceptance) / target_acceptance

        delta = jnp.where(flag, deltas_otherwise, delta)

        delta = jnp.clip(delta, min_delta, max_delta)
        carry_out = tuple(next_xs), delta, acceptance_rate, accepted_history
        return carry_out, None

    initial_accepted_history = jnp.zeros((window_size,)) * jnp.nan
    init_val = init_xs, initial_delta, 0., initial_accepted_history
    inps = jnp.arange(n_steps), jax.random.split(key, n_steps)
    (fin_xs, fin_delta, *_), _ = jax.lax.scan(body, init_val, inps)
    return fin_xs, fin_delta


def sampling_routine(key,
                     init_xs,
                     kernel,
                     n_steps,
                     verbose=False,
                     get_samples=True):
    if verbose:
        decorator = progress_bar_scan(n_steps)
    else:
        decorator = lambda x: x

    @decorator
    def body(xs, inp):
        i, key_op = inp

        # Run kernel
        *next_xs, accepted = kernel(key_op, xs)
        next_xs = tuple(next_xs)
        save = (next_xs, accepted) if get_samples else (None, accepted)
        return next_xs, save

    inps = jnp.arange(n_steps), jax.random.split(key, n_steps)
    final_xs, out = jax.lax.scan(body, init_xs, inps)
    if get_samples:
        xs_all, accepted_stats = out
        xs_all, *_ = xs_all  # remove auxiliary variables
        return xs_all[::int(get_samples)], accepted_stats[::int(get_samples)]
    else:
        _, accepted_stats = out
        return final_xs[0], accepted_stats
