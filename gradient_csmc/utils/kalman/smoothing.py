import jax
import jax.numpy as jnp
from chex import Array
from jax.scipy.linalg import solve


def smoothing(ms: Array, Ps: Array, Fs, Qs, bs) -> tuple[Array, Array]:
    """
    Samples from the pathwise smoothing distribution a LGSSM.

    Parameters
    ----------
    ms: Array
        Filtering means of the LGSSM.
    Ps: Array
        Filtering covariances of the LGSSM.
    Fs, Qs, bs: Array
        LGSSM model params.
    """

    def body(carry, inputs):
        m_t_p_1, P_t_p_1 = carry
        m_t, P_t, F_t, Q_t, b_t = inputs

        mean_diff = m_t_p_1 - (b_t + F_t @ m_t)
        S = F_t @ P_t @ F_t.T + Q_t
        cov_diff = P_t_p_1 - S

        gain = P_t @ solve(S, F_t, assume_a="pos").T
        m_t = m_t + gain @ mean_diff
        P_t = P_t + gain @ cov_diff @ gain.T

        return (m_t, P_t), (m_t, P_t)

    _, (ms_out, Ps_out) = jax.lax.scan(body, (ms[-1], Ps[-1]), (ms[:-1], Ps[:-1], Fs, Qs, bs), reverse=True)

    ms_out = jnp.append(ms_out, ms[-1][None, ...], 0)
    Ps_out = jnp.append(Ps_out, Ps[-1][None, ...], 0)
    return ms_out, Ps_out
