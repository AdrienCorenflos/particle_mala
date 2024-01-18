"""
Implements a multi-try IMH kernel with independent proposals.
"""
from functools import partial

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey

from gradient_csmc.utils.common import force_move
from gradient_csmc.utils.kalman import sampling
from gradient_csmc.utils.math import mvn_logpdf


def get_kernel(m0: Array, P0: Array,
               Fs, Qs, bs, log_pdf,
               N: int = 1,
               ):
    """
    Get an IMH kernel using the prior dynamics as proposal.

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
        IMH kernel function.
    """
    T, d_x, _ = Fs.shape
    T += 1

    def kernel(key: PRNGKey, state):
        """
        Implements a multi-try MH kernel with independent proposals.

        Parameters
        ----------
        key: PRNGKey
            JAX PRNGKey.
        state:
            Reference trajectory.

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

        x_star, val_x_star = state
        key_sample, key_choice, key_aux = jax.random.split(key, 3)

        samples = sampling.prior_sampling(key_sample, m0, P0, Fs, Qs, bs, N)
        log_weights = log_pdf(samples)

        samples = jnp.insert(samples, 0, x_star, axis=0)
        log_weights = jnp.insert(log_weights, 0, val_x_star, axis=0)

        weights = jnp.exp(log_weights - jnp.max(log_weights))
        weights /= jnp.sum(weights)

        idx, alpha = force_move(key_choice, weights, 0)
        x_star = samples[idx]
        val_x_star = log_weights[idx]

        return x_star, val_x_star, idx != 0

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
    val_x = log_pdf(x)
    return x, val_x


@partial(jnp.vectorize, signature='(t,d),(d),(d,d),(s,d,d),(s,d,d),(s,d)->()')
def prior_logpdf(xs, m0, inv_chol_P0, Fs, inv_chol_Qs, bs):
    out = mvn_logpdf(xs[0], m0, None, inv_chol_P0, constant=False)
    pred_xs = jnp.einsum('...j,...ij->...i', xs[:-1], Fs) + bs
    out += jnp.sum(mvn_logpdf(xs[1:], pred_xs, None, inv_chol_Qs, constant=False))
    return out
