"General log_phi function, mostly for testing purposes."""
from functools import partial

import jax.numpy as jnp
from jax.scipy.linalg import solve


def get_log_H(xs, ds, B, C_inv, E):
    N, D = xs.shape
    N = N - 1
    x_bar = jnp.mean(xs, 0)
    if len(ds.shape) == 1:
        ds = jnp.repeat(ds[None, :], N + 1, axis=0)
    d_bar = jnp.mean(ds, 0)

    @partial(jnp.vectorize, signature='(n),(n),(n)->()')
    def log_H(x, d, phi):
        Kappa = C_inv @ B @ E
        Nu = Kappa.T @ B @ E
        G = Kappa @ solve(E + N * Nu, Kappa.T, assume_a="pos")

        x_d = x - d
        B_x_phi = B @ (x + phi)

        term_1 = -jnp.dot(x_d, (C_inv + G) @ x_d)
        term_2 = N * jnp.dot(B_x_phi, jnp.dot(C_inv - N * G, B_x_phi))
        term_3 = 2 * (N + 1) * jnp.dot(x_bar - d_bar, G @ x_d - (C_inv - N * G) @ B_x_phi)
        term_4 = 2 * jnp.dot(x_d, (C_inv - N * G) @ B_x_phi)

        out = -0.5 * (term_1 + term_2 + term_3 + term_4)
        return jnp.nan_to_num(out)

    return log_H
