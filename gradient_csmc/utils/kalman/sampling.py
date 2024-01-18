import jax
import jax.numpy as jnp
from chex import Array, PRNGKey

from gradient_csmc.utils.kalman.filtering import sequential_update


def sampling(key: PRNGKey, ms: Array, Ps: Array, Fs, Qs, bs, N=1) -> Array:
    """
    Samples from the pathwise smoothing distribution a LGSSM.

    Parameters
    ----------
    key: PRNGKey
        Random number generator key.
    ms: Array
        Filtering means of the LGSSM.
    Ps: Array
        Filtering covariances of the LGSSM.
    Fs, Qs, bs: Array
        LGSSM model params.
    N: int
        Number of samples to draw.
    """

    epsilons = jax.random.normal(key, shape=ms.shape + (N,))

    def body(x_t_p_1, inputs):
        eps_t, m_t, P_t, F_t, Q_t, b_t = inputs
        m_t, P_t, _ = sequential_update(x_t_p_1, m_t, P_t, F_t, b_t, Q_t)
        chol_P_t = jnp.linalg.cholesky(P_t)
        inc_t = jnp.einsum("...ij,j...->...i", chol_P_t, eps_t)
        inc_t = jnp.nan_to_num(inc_t)
        x_t = m_t + inc_t
        return x_t, x_t

    chol_P_T = jnp.linalg.cholesky(Ps[-1])
    inc_T = jnp.nan_to_num(chol_P_T @ epsilons[0])
    x_T = jnp.expand_dims(ms[-1], 1) + inc_T
    x_T = x_T.T
    _, samples = jax.lax.scan(body, x_T, (epsilons[1:], ms[:-1], Ps[:-1], Fs, Qs, bs), reverse=True)
    samples = jnp.append(samples, x_T[None, ...], 0)
    samples = jnp.transpose(samples, (1, 0, 2))
    return samples


def prior_sampling(key: PRNGKey, m0, P0, Fs, Qs, bs, N=1) -> Array:
    """
    Samples from Gaussian dynamics.

    Parameters
    ----------
    key: PRNGKey
        Random number generator key.
    m0: Array
        Initial mean.
    P0: Array
        Initial covariance.
    Fs, Qs, bs: Array
        Dynamics model params.
    N: int
        Number of samples to draw.
    """
    T, d_x, _ = Fs.shape
    T += 1
    epsilons = jax.random.normal(key, shape=(T, d_x, N,))
    x0 = m0[:, None] + jnp.linalg.cholesky(P0) @ epsilons[0]

    def body(x_t_m_1, inputs):
        eps_t, F_t, Q_t, b_t = inputs
        x_t = F_t @ x_t_m_1 + b_t[:, None] + jnp.linalg.cholesky(Q_t) @ eps_t
        return x_t, x_t

    _, samples = jax.lax.scan(body, x0, (epsilons[1:], Fs, Qs, bs))

    samples = jnp.insert(samples, 0, x0[None, ...], 0)
    samples = jnp.transpose(samples, (2, 0, 1))
    return samples
