from typing import Any

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey


def force_move(key: PRNGKey, weights: Array, k: int) -> [Array, float]:
    """
    Forced-move trajectory selection. The weights are assumed to be normalised already.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Log-weights of the particles.
    k:
        Index of the reference particle.

    Returns
    -------
    l_T:
        New index of the ancestor of the reference particle.
    alpha:
        Probability of accepting new sample.
    """
    # TODO: log space?
    M = weights.shape[0]
    key_1, key_2 = jax.random.split(key, 2)

    w_k = weights[k]
    temp = 1 - w_k

    rest_weights = weights.at[k].set(0)  # w_{-k}
    threshold = jnp.maximum(1 - jnp.exp(-M), 1 - 1e-12)
    rest_weights = jax.lax.cond(w_k < threshold, lambda: rest_weights / temp,
                                lambda: jnp.full((M,), 1 / M))  # w_{-k} / (1 - w_k)

    i = jax.random.choice(key_1, M, p=rest_weights, shape=())  # i ~ Cat(w_{-k} / (1 - w_k))
    u = jax.random.uniform(key_2, shape=())
    accept = u * (1 - weights[i]) < temp  # u < (1 - w_k) / (1 - w_i)

    alpha = jnp.nansum(temp * rest_weights / (1 - weights))
    i = jax.lax.select(accept, i, k)

    return i, jnp.clip(alpha, 0, 1.)


def barker_move(key: PRNGKey, weights: Array, k: Any = 0) -> [Array, float]:
    """
    Forced-move trajectory selection. The weights are assumed to be normalised already.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Log-weights of the particles.
    k:
        Index of the reference particle. Not used for the Barker move.

    Returns
    -------
    l_T:
        New index of the ancestor of the reference particle.
    alpha:
        Probability of accepting new sample.
    """
    M = weights.shape[0]
    i = jax.random.choice(key, M, p=weights, shape=())
    return i, 1 - weights[k]


def _log_1_m_exp(arr):
    """ Computes log(1 - exp(arr)) in a numerically stable way. """
    max_arr = jnp.max(arr)
    return jnp.log(-jnp.expm1(arr - max_arr)) + max_arr


def test_uniform():
    import numpy.testing as npt
    import numpy as np
    M = 1_000_000
    K = 8
    ws = jnp.ones(K) / K
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, M)

    forced_out, forced_alphas = jax.vmap(lambda k: force_move(k, ws, 0))(keys)
    barker_out, barker_alphas = jax.vmap(lambda k: barker_move(k, ws, 0))(keys)

    forced_count = np.unique(forced_out, return_counts=True)
    barker_count = np.unique(barker_out, return_counts=True)
    npt.assert_allclose(forced_count[1], M / (K - 1), rtol=1e-2)
    npt.assert_allclose(barker_count[1], M / K, rtol=1e-2)
    npt.assert_allclose(forced_alphas, 1., rtol=1e-2)
    npt.assert_allclose(barker_alphas, (K - 1) / K, rtol=1e-2)


def test_degenerate():
    import numpy.testing as npt

    M = 1_000_000
    K = 8
    ws = jnp.zeros(K)
    ws = ws.at[0].set(1.)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, M)

    forced_out, forced_alphas = jax.vmap(lambda k: force_move(k, ws, 0))(keys)
    barker_out, barker_alphas = jax.vmap(lambda k: barker_move(k, ws, 0))(keys)

    npt.assert_allclose(forced_out, 0)
    npt.assert_allclose(barker_out, 0)
    npt.assert_allclose(forced_alphas, 0., rtol=1e-2)
    npt.assert_allclose(barker_alphas, 0., rtol=1e-2)

    forced_out, forced_alphas = jax.vmap(lambda k: force_move(k, ws, 1))(keys)
    barker_out, barker_alphas = jax.vmap(lambda k: barker_move(k, ws, 1))(keys)

    npt.assert_allclose(forced_out, 0)
    npt.assert_allclose(barker_out, 0)
    npt.assert_allclose(forced_alphas, 1., rtol=1e-2)
    npt.assert_allclose(barker_alphas, 1., rtol=1e-2)


def test_two_samples():
    import numpy.testing as npt
    import numpy as np

    M = 1_000_000
    ws = jnp.array([0.1, 0.9])
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, M)

    forced_out, forced_alphas = jax.vmap(lambda k: force_move(k, ws, 0))(keys)
    barker_out, barker_alphas = jax.vmap(lambda k: barker_move(k, ws, 0))(keys)

    barker_count = np.unique(barker_out, return_counts=True)
    npt.assert_allclose(forced_out, 1, rtol=1e-2)
    npt.assert_array_almost_equal(barker_count[1] / M, np.array([0.1, 0.9]), decimal=3)
    npt.assert_allclose(forced_alphas, 1., rtol=1e-2)
    npt.assert_allclose(barker_alphas, 0.9, rtol=1e-2)

    forced_out, forced_alphas = jax.vmap(lambda k: force_move(k, ws, 1))(keys)
    barker_out, barker_alphas = jax.vmap(lambda k: barker_move(k, ws, 1))(keys)

    forced_count = np.unique(forced_out, return_counts=True)
    barker_count = np.unique(barker_out, return_counts=True)

    npt.assert_array_almost_equal(forced_count[1] / M, np.array([1 / 9, 8 / 9]), decimal=3)
    npt.assert_array_almost_equal(barker_count[1] / M, np.array([0.1, 0.9]), decimal=3)

    npt.assert_allclose(forced_alphas, 1 / 9, rtol=1e-2)
    npt.assert_allclose(barker_alphas, 1 / 10, rtol=1e-2)
