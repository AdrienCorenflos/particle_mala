import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax.scipy.stats import multivariate_normal as mvn

from gradient_csmc.utils.marginalisation import get_log_H


@pytest.fixture(scope='module', autouse=True)
def setup():
    jax.config.update("jax_enable_x64", True)


def q_full(xs, ds, B, C, E, phis, n):
    N, D = xs.shape

    eye = jnp.eye(N - 1)
    one = jnp.ones((N - 1, 1))

    Bs = jnp.kron(one, B)
    Cs = jnp.kron(eye, C)
    ds_n = jnp.delete(ds, n, axis=0)

    mu = ds_n.reshape(-1) + Bs @ (xs[n] + phis[n])
    Sigma = Cs + Bs @ E @ Bs.T

    xs = jnp.delete(xs, n, 0).reshape(-1)
    return mvn.logpdf(xs, mu, Sigma)


@pytest.mark.parametrize('seed', [0, 1, 2, 42, 666, 31415])
def test(seed):
    np.random.seed(seed)

    N_SAMPLES = 10
    DIM = 5

    XS = np.random.randn(N_SAMPLES, DIM)
    DS = np.random.randn(N_SAMPLES, DIM)
    PHIS = np.random.randn(N_SAMPLES, DIM)
    B_ = np.random.randn(DIM, DIM)

    C_ = np.random.randn(DIM, DIM ** 2)
    C_ = C_ @ C_.T
    C_inv = np.linalg.pinv(C_, hermitian=True)

    E_ = np.random.randn(DIM, DIM ** 2)
    E_ = E_ @ E_.T

    log_H = get_log_H(XS, DS, B_, C_inv, E_)

    i, j = np.random.choice(N_SAMPLES, 2, replace=False)

    actual = log_H(XS[i], DS[i], PHIS[i]) - log_H(XS[j], DS[j], PHIS[j])
    expected = q_full(XS, DS, B_, C_, E_, PHIS, i) - q_full(XS, DS, B_, C_, E_, PHIS, j)
    # Compare ratios
    npt.assert_almost_equal(actual, expected, decimal=10)
