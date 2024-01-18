"""
Implements the twisted Particle-aGrad kernel of the paper.
"""
from functools import partial
from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp
from jax.scipy.linalg import solve
from jax.scipy.linalg import solve_triangular

from gradient_csmc.csmc import kernel as csmc_kernel
from gradient_csmc.utils.kalman.filtering import filtering
from gradient_csmc.utils.math import mvn_logpdf


def get_kernel(mu0: Array, P0: Array, r0: Callable,
               Fs, bs, Qs,
               rt: Union[Callable, tuple[Callable, Any]],
               resampling_func: Callable, ancestor_move_func: Callable, N: int,
               backward: bool = False
               ):
    """
    Constructor for the twisted Particle-aGrad kernel.

    Parameters
    ----------
    mu0:
        Initial distribution mean vector.
    P0:
        Initial distribution covariance matrix.
    r0:
        Initial distribution log-density.
    Fs, bs, Qs:
        Array of parameters matrices for transition functions: p(xt | xt-1) = N(xt | Fs[t-1] xt-1 + bs[t-1] , Qs[t-1]).
    rt:
        Potential function. Either a callable or a tuple of a callable and its parameters.
    resampling_func:
        Resampling function.
    ancestor_move_func:
        Ancestor move function.
    N:
        Number of particles.
    backward:
        Whether to run the backward sampling kernel.

    Returns
    -------
    kernel: Callable
        ATP-CSMC kernel function.

    """
    d_x, T = mu0.shape[0], Fs.shape[0] + 1

    rev_Hs = jnp.repeat(jnp.eye(d_x)[None, :, :], T, 0)
    rev_cs = jnp.zeros((T, d_x))

    # Compute the prior mean and covariance matrices
    mT, PT, rev_Fs, rev_Qs, rev_bs = _get_reversed_dynamics(mu0, P0, Fs, Qs, bs)

    rt, rt_params = rt if isinstance(rt, tuple) else (rt, None)

    val_and_grad_r0 = jnp.vectorize(jax.value_and_grad(r0), signature='(d)->(),(d)')
    val_and_grad_rt = jnp.vectorize(jax.value_and_grad(rt, argnums=1), signature='(d),(d)->(),(d)', excluded=(2,))

    def kernel(key: PRNGKey, x_star: Array, b_star: Array, ells: Array, deltas: Array):
        """
        Implements the twisted Particle-aGrad kernel of the paper for pre-computed reversed dynamics.

        Parameters
        ----------
        key: PRNGKey
            JAX PRNGKey.
        x_star:
            Reference trajectory.
        b_star:
            Reference ancestor indices.
        ells:
            Proposal variance scaling.
        deltas:
            Gradient step-size.

        Returns
        -------
        x_star: Array
            New reference trajectory.
        b_star: Array
            New reference ancestor indices.

        """
        ###############################
        #        HOUSEKEEPING         #
        ###############################

        key_csmc, _, key_aux = jax.random.split(key, 3)
        aux_std_devs = jnp.sqrt(0.5 * ells)

        _, grad_log_w_star_0 = val_and_grad_r0(x_star[0])
        _, grad_log_w_star = jax.vmap(val_and_grad_rt)(x_star[:-1], x_star[1:], rt_params)
        grad_log_w_star = jnp.insert(grad_log_w_star, 0, grad_log_w_star_0, axis=0)

        eps_aux = jax.random.normal(key_aux, shape=(T, d_x))
        aux_vars = x_star + 0.5 * deltas[:, None] * grad_log_w_star + aux_std_devs[:, None] * eps_aux

        ###############################################
        #       Proposal and weight functions         #
        ###############################################

        ### Kalman stuff
        # Marginal posteriors
        rev_Rs = aux_std_devs[:, None, None] ** 2 * jnp.eye(d_x)[None, :, :]
        t_ms_u, t_Ps_u, _ = filtering(aux_vars[::-1], mT, PT, rev_Fs[::-1], rev_Qs[::-1], rev_bs[::-1], rev_Hs,
                                      rev_Rs[::-1], rev_cs)  # p(x_t | u_{t:T})
        t_ms_u, t_Ps_u = t_ms_u[::-1], t_Ps_u[::-1]

        As, cs, chol_Es = jax.vmap(make_proposal)(t_ms_u[1:], t_Ps_u[1:], rev_Fs, rev_bs, rev_Qs)
        inv_chol_Es = jax.vmap(lambda z: solve_triangular(z, jnp.eye(z.shape[-1]), lower=True))(chol_Es)

        t_m0_u, t_P0_u = t_ms_u[0], t_Ps_u[0]
        chol_t_P0_u = jnp.linalg.cholesky(t_P0_u)
        inv_chol_t_P0_u = solve_triangular(chol_t_P0_u, jnp.eye(t_P0_u.shape[-1]), lower=True)

        # jax.debug.print("aux_vars: {}", aux_vars)
        # jax.debug.print("t_m0_u: {}", t_m0_u)
        # jax.debug.print("t_P0_u: {}", t_P0_u)

        spec_M0_rvs = partial(M0_rvs, params=(t_m0_u, chol_t_P0_u))
        spec_M0_logpdf = partial(M0_logpdf, params=(t_m0_u, inv_chol_t_P0_u))

        spec_Gamma_0 = partial(Gamma_0, val_and_grad_r0=val_and_grad_r0, params=(aux_vars[0],
                                                                                 t_m0_u, inv_chol_t_P0_u,
                                                                                 deltas[0], ells[0]))
        spec_Gamma_t = partial(Gamma_t, val_and_grad_rt=val_and_grad_rt)
        Mt_params = As, cs, chol_Es, inv_chol_Es
        Gamma_t_params = aux_vars[1:], deltas[1:], ells[1:], rt_params, Mt_params

        Mt = Mt_rvs, Mt_logpdf, Mt_params
        Gamma_t_plus_params = spec_Gamma_t, Gamma_t_params
        M0 = spec_M0_rvs, spec_M0_logpdf

        ###########################
        #       Call CSMC         #
        ###########################
        return csmc_kernel(key_csmc, x_star, b_star, M0, spec_Gamma_0, Mt, Gamma_t_plus_params, resampling_func,
                           ancestor_move_func, N, backward)

    return kernel


def make_proposal(t_m_u, t_P_u, rev_F, rev_b, rev_Q):
    """
    Compute the parameters A, c, E of p(x_{t+1} | x_t, u_{t+1:T}) = N(x_{t+1} | A x_t + c, E).

    Parameters
    ----------
    t_m_u, t_P_u: Array
        Mean and covariance of p(x_{t+1} | u_{t+1:T})
    rev_F, rev_b, rev_Q: Array
        Parameters for the reverse transition model p(x_t | x_{t+1})

    Returns
    -------
    A, c, E: Array
        Parameters of the proposal distribution.
    """

    S = rev_Q + rev_F @ t_P_u @ rev_F.T
    A = solve(S, rev_F @ t_P_u, assume_a="pos").T
    rev_x_hat = rev_F @ t_m_u + rev_b

    c = t_m_u - A @ rev_x_hat
    E = t_P_u - A @ S @ A.T
    E = 0.5 * (E + E.T)
    return A, c, jnp.linalg.cholesky(E)


def M0_rvs(key, N, params):
    m, cholP = params
    dx = m.shape[0]
    eps = jax.random.normal(key, (N, dx))
    return m[None, :] + eps @ cholP.T


def M0_logpdf(x, params):
    m, inv_cholP = params
    return mvn_logpdf(x, m, None, inv_cholP, constant=False)


def Mt_rvs(key, x_prev, params):
    A, c, cholE, _ = params
    N_, dx = x_prev.shape
    mean_t = x_prev @ A.T + c[None, :]
    eps = jax.random.normal(key, (N_, dx))
    return mean_t + eps @ cholE.T


def Mt_logpdf(x_prev, x, params):
    A, c, _, inv_cholE = params
    mean_t = x_prev @ A.T + c[None, :]
    return mvn_logpdf(x, mean_t, None, inv_cholE, constant=False)


def Gamma_0(x, val_and_grad_r0, params):
    u, prop_m0, prop_inv_chol_P0, delta, ell = params
    r0_val, r0_grad = val_and_grad_r0(x)
    out = r0_val
    correction_term = -jnp.sum((u - x - 0.5 * delta * r0_grad) ** 2, -1) / ell
    correction_term += jnp.sum((u - x) ** 2, -1) / ell
    correction_term += mvn_logpdf(x, prop_m0, None, prop_inv_chol_P0, False)
    out += correction_term
    return out


def Gamma_t(x_prev, x, params, val_and_grad_rt):
    u, delta, ell, rt_params_here, Mt_params_here = params
    A, c, chol_E, inv_chol_E = Mt_params_here

    rt_val, rt_grad = val_and_grad_rt(x_prev, x, rt_params_here)
    out = rt_val
    correction_term = -jnp.sum((u - x - 0.5 * delta * rt_grad) ** 2, -1) / ell
    correction_term += jnp.sum((u - x) ** 2, -1) / ell
    correction_term += mvn_logpdf(x, x_prev @ A.T + c[None, :], None, chol_inv=inv_chol_E, constant=False)
    out += correction_term
    return out


def delta_adaptation_routine(
        key,
        init_xs, init_bs,
        kernel,
        target_acceptance,
        initial_delta,
        n_steps,
        verbose=False,
        min_delta=1e-12,
        max_delta=1e2,
        min_rate=1e-2,
        window_size=100,
        rate=0.1,
        target_stat: Union[str, float] = "mean",
        **_kwargs
):
    T = init_xs.shape[0]

    if verbose:
        from gradient_csmc.utils.pbar import progress_bar_scan
        decorator = progress_bar_scan(n_steps, show=-1)
    else:
        decorator = lambda x: x

    @decorator
    def body(carry, inp):
        state, deltas, accepted_history, _ = carry
        xs, bs = state
        i, key_i = inp

        # Run kernel
        next_xs, next_bs, *_ = kernel(key_i, state, deltas)

        accepted = next_bs != bs
        accepted_history = accepted_history.at[1:, :].set(accepted_history[:-1])
        accepted_history = accepted_history.at[0, :].set(accepted)
        accepted_history_mean = jnp.nanmean(accepted_history, 0)

        if target_stat == "mean":
            acceptance_rate = jnp.nanmean(accepted_history_mean)
        elif target_stat == "median":
            acceptance_rate = jnp.nanmedian(accepted_history_mean)
        elif target_stat == "max":
            acceptance_rate = jnp.max(accepted_history_mean)
        elif target_stat == "min":
            acceptance_rate = jnp.min(accepted_history_mean)
        else:
            acceptance_rate = jnp.quantile(accepted_history_mean, target_stat)

        flag = jnp.logical_or(acceptance_rate < target_acceptance - 0.05,
                              acceptance_rate > target_acceptance + 0.05)
        flag &= i > window_size
        rate_i = jnp.maximum(min_rate, rate / (i + 1) ** 0.5)

        deltas_otherwise = deltas + rate_i * deltas * (
                acceptance_rate - target_acceptance) / target_acceptance

        deltas = jnp.where(flag, deltas_otherwise, deltas)

        deltas = jnp.clip(deltas, min_delta, max_delta)
        carry_out = (next_xs, next_bs), deltas, accepted_history, acceptance_rate * 1.
        return carry_out, None

    initial_delta = initial_delta * jnp.ones((T,))
    initial_accepted_history = jnp.zeros((window_size, T)) * jnp.nan
    init = (init_xs, init_bs), initial_delta, initial_accepted_history, 0.
    inps = jnp.arange(n_steps), jax.random.split(key, n_steps)
    (fin_state, fin_deltas, *_), _ = jax.lax.scan(body, init, inps)
    return fin_state, fin_deltas


def _get_reversed_dynamics(m0, P0, Fs, Qs, bs):
    """
    Computes the reversed dynamics of the state-space model.
    """

    _, d = bs.shape

    def body(carry, inp):
        prev_m, prev_P = carry
        F, Q, b = inp

        K = _get_gain(F, Q, prev_P)
        Sigma = prev_P - K @ F @ prev_P
        Sigma = 0.5 * (Sigma + Sigma.T)
        mu = prev_m - K @ (F @ prev_m + b)

        m = F.T @ prev_m - F.T @ b
        P = F.T @ prev_P @ F + Q
        return (m, 0.5 * (P + P.T)), (K, Sigma, mu)

    (m_T, P_T), (Ks, Sigmas, mus) = jax.lax.scan(body, (m0, P0), (Fs, Qs, bs))
    return m_T, P_T, Ks, Sigmas, mus


def _get_gain(F, Q, P):
    return solve(F @ P @ F.T + Q, P @ F.T, assume_a="pos").T


def test_get_reverse_dynamics():
    import numpy.testing as npt
    # test that stationary dynamics are left unchanged
    m0 = jnp.zeros((2,))
    P0 = jnp.eye(2)

    Fs = jnp.array([[[0.75, 0.], [0., 0.75]]])
    Qs = jnp.eye(2) - jnp.einsum("...ij,...jk", Fs, Fs)
    bs = jnp.zeros((1, 2))

    mT, PT, rev_Fs, rev_Qs, rev_bs = _get_reversed_dynamics(m0, P0, Fs, Qs, bs)

    npt.assert_allclose(mT, m0)
    npt.assert_allclose(PT, P0)

    npt.assert_allclose(rev_Fs, Fs)
    npt.assert_allclose(rev_Qs, Qs)
    npt.assert_allclose(rev_bs, bs)
