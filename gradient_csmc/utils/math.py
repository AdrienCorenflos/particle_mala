import math
from functools import partial

import jax
from chex import PRNGKey
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.stats import norm


def mvn_logpdf(x, m, chol, chol_inv=None, constant=True):
    """
    Computes the log of the probability density function of a multivariate normal distribution with mean m and
    covariance matrix chol @ chol.T. Can either use the Cholesky factor chol or its inverse chol_inv.

    Parameters
    ----------
    x: Array
        Point where the density is evaluated.
    m: Array
        Mean of the distribution.
    chol: Array
        Lower triangular Cholesky factor of the covariance matrix.
    chol_inv: Array, optional
        Inverse of the Cholesky factor of the covariance matrix.
    constant: bool, optional
        Whether to return the log density with respect to the Lebesgue measure or with respect to the Gaussian measure.

    Returns
    -------
    logpdf: float
        Log of the probability density function evaluated at x.
    """
    # Numerically ignore nans and infs

    if chol_inv is not None:
        out = _logpdf_with_inv(x, m, chol_inv)
    else:
        out = _logpdf(x, m, chol)
    if constant:
        normalizing_constant = _get_constant(chol)
    else:
        normalizing_constant = 0.

    return out - normalizing_constant


@partial(jnp.vectorize, signature="(n,n)->()")
def _get_constant(chol):
    chol_diag = jnp.diag(chol)
    dim = jnp.sum(jnp.isfinite(chol_diag))
    return tril_log_det(chol) + 0.5 * dim * math.log(2 * math.pi)


def tril_log_det(chol):
    # Replace nans and infs in the Cholesky decomposition by 1. as they will then be ignored by the log.
    if jnp.ndim(chol) == 2:
        diag_chol = jnp.nan_to_num(jnp.diag(chol), nan=1., posinf=1., neginf=1.)
    else:
        diag_chol = jnp.nan_to_num(chol, nan=1., posinf=1., neginf=1.)
    return jnp.nansum(jnp.log(jnp.abs(diag_chol)))


def rvs(key: PRNGKey, m, chol):
    """
    Samples from the multivariate normal distribution.

    Parameters
    ----------
    key: PRNGKey
        Random number generator key.
    m: Array
        Mean of the multivariate normal distribution.
    chol: Array
        Cholesky decomposition of the covariance matrix of the multivariate normal distribution.
    """
    eps = jax.random.normal(key, shape=m.shape)
    return m + jnp.einsum("...ij,...j->...i", chol, eps)


def norm_logpdf(x, loc, scale, constant=True):
    if constant:
        return norm.logpdf(x, loc, scale)
    z = (x - loc) / scale
    return -0.5 * z ** 2


@partial(jnp.vectorize, signature="(n),(n),(n,n)->()")
def _logpdf_with_inv(x, m, chol_inv):
    chol_inv_clip = jnp.nan_to_num(chol_inv, nan=0., posinf=0., neginf=0.)
    y = chol_inv_clip @ (x - m)
    norm_y = jnp.sum(y * y)
    return -0.5 * norm_y


@partial(jnp.vectorize, signature="(n),(n),(n,n)->()")
def _logpdf(x, m, chol):
    y = solve_triangular(chol, x - m, lower=True)
    norm_y = jnp.sum(y * y)
    return -0.5 * norm_y
