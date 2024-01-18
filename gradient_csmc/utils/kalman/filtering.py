from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from chex import Numeric, Array
from jax.scipy.linalg import cho_solve
from jax.tree_util import tree_map

from ..math import mvn_logpdf


def filtering(ys, m0, P0, Fs, Qs, bs, Hs, Rs, cs) -> Tuple[Array, Array, Numeric]:
    """
    Kalman filtering algorithm.
    If the number of observations is equal to the number of states, the first observation is used to update the initial state.
    Otherwise, the initial state is not updated but propagated first.

        Parameters
        ----------
        ys : Array
            Observations of shape (T, D_y).
        m0, P0, Fs, Qs, bs, Hs, Rs, cs: Array
            LGSSM parameters.

        Returns
        -------
        ms : Array
            Filtered state means.
        Ps : Array
            Filtered state covariances.
        ell : Numeric
            Log-likelihood of the observations.
        """

    ny, nx = ys.shape[0], bs.shape[0] + 1

    ell0 = 0.
    if nx == ny:
        # split between initial observation and rest
        (y0, ys), (H0, Hs), (c0, cs), (R0, Rs) = jax.tree_map(lambda x: (x[0], x[1:]), (ys, Hs, cs, Rs))
        # Update initial state
        m0, P0, ell0 = sequential_update(y0, m0, P0, H0, c0, R0)

    def body(carry, inputs):
        m, P, curr_ell = carry
        F, Q, b, H, R, c, y = inputs
        m, P, ell_inc = sequential_predict_update(m, P, F, b, Q, y, H, c, R)
        return (m, P, curr_ell + ell_inc), (m, P)

    (*_, ell), (ms, Ps) = jax.lax.scan(body,
                                       (m0, P0, ell0),
                                       (Fs, Qs, bs, Hs, Rs, cs, ys))

    ms, Ps = tree_map(lambda z, y: jnp.insert(z, 0, y, axis=0), (ms, Ps), (m0, P0))
    return ms, Ps, ell


#                                   y,    m,     P,     H,    c,    R,  ->  m,     P,  ell
@partial(jnp.vectorize, signature='(dy),(dx),(dx,dx),(dy,dx),(dy),(dy,dy)->(dx),(dx,dx),()')
def sequential_update(y, m, P, H, c, R):
    y_hat = H @ m + c
    y_diff = y - y_hat

    S = R + H @ P @ H.T

    chol_S = jnp.linalg.cholesky(S)
    ell_inc = mvn_logpdf(y, y_hat, chol_S)
    G = cho_solve((chol_S, True), H @ P).T

    m = m + G @ y_diff

    P = P - G @ S @ G.T
    P = 0.5 * (P + P.T)
    return m, P, jnp.nan_to_num(ell_inc, nan=0.)


#                                   m,     P,      F,     b,    Q,  ->  m,    P,
@partial(jnp.vectorize, signature='(dx),(dx,dx),(dx,dx),(dx),(dx,dx)->(dx),(dx,dx)')
def sequential_predict(m, P, F, b, Q):
    m = F @ m + b
    P = Q + F @ P @ F.T
    P = 0.5 * (P + P.T)
    return m, P


#                                   m,     P,      F,     b,    Q,     y,    H,     c,    R   ->  m,    P,   ell
@partial(jnp.vectorize, signature='(dx),(dx,dx),(dx,dx),(dx),(dx,dx),(dy),(dy,dx),(dy),(dy,dy)->(dx),(dx,dx),()')
def sequential_predict_update(m, P, F, b, Q, y, H, c, R):
    m, P = sequential_predict(m, P, F, b, Q)
    m, P, ell_inc = sequential_update(y, m, P, H, c, R)
    return m, P, ell_inc
