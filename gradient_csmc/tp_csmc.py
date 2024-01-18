"""
Implements the Particle-mGrad kernel of the paper.
"""
from functools import partial
from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular, solve

from gradient_csmc.atp_csmc_f import make_proposal
from gradient_csmc.csmc import kernel as csmc_kernel
from gradient_csmc.utils.math import mvn_logpdf


def get_kernel(mu0: Array, P0: Array, r0: Callable,
               mut: Union[Callable, tuple[Callable, Any]], Qs,
               rt: Union[Callable, tuple[Callable, Any]],
               resampling_func: Callable, ancestor_move_func: Callable, N: int,
               backward: bool = False
               ):
    """
    Constructor for the Particle-mGrad kernel.

    Parameters
    ----------
    mu0:
        Initial distribution mean.
    P0:
        Initial distribution covariance matrix.
    r0:
        Initial weight function.
    mut:
        If a tuple, the first element is the function and the second element is the parameters.
        mut(x_prev, params) returns the mean of the transition function at time t.
    rt:
        If a tuple, the first element is the function and the second element is the parameters.
        rt(x_prev, x, params) returns the log-likelihood at time t.
    Qs:
        Array of covariance matrices. The first element is the covariance at time 0: p(x0) = N(m0, Qs[0]).
        The rest is the covariance matrix of transition functions: p(xt | xt-1) = N(xt | µ(xt-1), Qs[t]).
    resampling_func:
        Resampling scheme to use.
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
    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Qs = jnp.linalg.cholesky(Qs)

    inv_chol_P0 = solve_triangular(chol_P0, jnp.eye(P0.shape[-1]), lower=True)
    inv_chol_Qs = jax.vmap(lambda z: solve_triangular(z, jnp.eye(Qs.shape[-1]), lower=True))(chol_Qs)

    Qs_ = jnp.insert(Qs, 0, P0, axis=0)
    get_proposal_params = make_proposal(Qs_, get_K_inv=True)

    mut, mut_params = mut if isinstance(mut, tuple) else (mut, None)
    rt, rt_params = rt if isinstance(rt, tuple) else (rt, None)

    val_and_grad_r0 = jnp.vectorize(jax.value_and_grad(r0), signature='(d)->(),(d)')
    val_and_grad_rt = jnp.vectorize(jax.value_and_grad(rt, argnums=1), signature='(d),(d)->(),(d)', excluded=(2,))

    def kernel(key: PRNGKey, x_star: Array, b_star: Array, ells: Array, deltas: Array):
        """
        Implements the Particle-mGrad kernel of the paper for pre-computed parameters.

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

        Ks_1, Ks_2, chols_prop, chols_inv_prop, Ks_inv = get_proposal_params(ells)
        # We need these for the marginalisation, so it's a tiny tiny bit less efficient than the auxiliary version
        Ks = jnp.einsum('...ij,...jk->...ik', Ks_1, Ks_2)
        # Qs_inv_prop = jnp.einsum('...ij,...kj->...ik', chols_inv_prop, chols_inv_prop)

        T, d_x = x_star.shape
        key_csmc, key_aux = jax.random.split(key)
        aux_std_devs = jnp.sqrt(0.5 * ells)

        _, grad_log_w_star_0 = val_and_grad_r0(x_star[0])
        _, grad_log_w_star = jax.vmap(val_and_grad_rt)(x_star[:-1], x_star[1:], rt_params)
        grad_log_w_star = jnp.insert(grad_log_w_star, 0, grad_log_w_star_0, axis=0)

        eps_aux = jax.random.normal(key_aux, shape=(T, d_x))
        aux_vars = x_star + 0.5 * deltas[:, None] * grad_log_w_star + aux_std_devs[:, None] * eps_aux

        # eye = jnp.eye(d_x)

        ###############################################
        #       Proposal and weight functions         #
        ###############################################

        # Note that the logpdf of Mt and M0 are not really their logpdf, but rather the log_H corresponding to them.
        # This "hack" makes it easier to simply then apply CSMC to the problem.

        def M0_rvs(key_, N_):
            dx = mu0.shape[0]
            mean_0 = mu0 + Ks[0] @ (aux_vars[0] - mu0)
            eps = jax.random.normal(key_, (N_, dx))
            return mean_0[None, :] + eps @ chols_prop[0].T

        def M0_logpdf(x):
            ds = mu0 - Ks[0] @ mu0
            # log_phi_func = get_log_H(x, ds, Ks[0], Qs_inv_prop[0], 0.5 * ells[0] * eye)
            log_phi_func = get_log_H_bis(x, ds, Ks[0], Ks_inv[0], ells[0])
            _, grad_r0 = val_and_grad_r0(x)
            out = log_phi_func(x, ds, 0.5 * deltas[0] * grad_r0)
            return -out

        def Mt_rvs(key_, x_prev, params):
            (u, K, chol, *_), mu_params, _ = params
            N_, dx = x_prev.shape
            mean_prior = mut(x_prev, mu_params)
            mean_t = mean_prior + (u[None, :] - mean_prior) @ K.T
            eps = jax.random.normal(key_, (N_, dx))
            return mean_t + eps @ chol.T

        def Mt_logpdf(x_prev, x, params):
            (_, K, chol, K_inv, ell, delta), mu_params, r_params_here = params

            mean_prior = mut(x_prev, mu_params)
            ds = mean_prior - mean_prior @ K.T

            # log_phi_func = get_log_H(x, ds, K, Q_inv, 0.5 * ell * eye)
            log_phi_func = get_log_H_bis(x, ds, K, K_inv, ell)
            _, grad_r = val_and_grad_rt(x_prev, x, r_params_here)
            out = log_phi_func(x, ds, 0.5 * delta * grad_r)
            return -out

        def Gamma_0(x):
            out = mvn_logpdf(x, mu0, None, chol_inv=inv_chol_P0, constant=False)
            r0_val = r0(x)
            out += r0_val
            return out

        def Gamma_t(x_prev, x, params):
            inv_chol_Q, mu_params, rt_params_here = params

            out = mvn_logpdf(x, mut(x_prev, mu_params), None, chol_inv=inv_chol_Q, constant=False)

            rt_val = rt(x_prev, x, rt_params_here)
            out += rt_val
            return out

        Mt_params = (aux_vars[1:], Ks[1:], chols_prop[1:], Ks_inv[1:], ells[1:], deltas[1:]), mut_params, rt_params
        Mt = Mt_rvs, Mt_logpdf, Mt_params
        Gamma_t_params = inv_chol_Qs, mut_params, rt_params
        Gammat_plus_params = Gamma_t, Gamma_t_params
        M0 = M0_rvs, M0_logpdf

        ###############################################
        #       Proposal and weight functions         #
        ###############################################

        return csmc_kernel(key_csmc, x_star, b_star, M0, Gamma_0, Mt, Gammat_plus_params, resampling_func,
                           ancestor_move_func, N, backward)

    return kernel


def get_log_H_bis(xs, ds, K, K_inv, delta):
    N, D = xs.shape
    eye = jnp.eye(D)

    N = N - 1
    x_bar = jnp.mean(xs, 0)
    if len(ds.shape) == 1:
        ds = jnp.repeat(ds[None, :], N + 1, axis=0)
    d_bar = jnp.mean(ds, 0)
    x_d_bar = x_bar - d_bar

    G = 2 * solve(eye + N * K, eye, assume_a="pos") / delta

    @partial(jnp.vectorize, signature='(n),(n),(n)->()')
    def log_H(x, d, phi):
        x_d = x - d

        term_1 = -jnp.dot(x_d, (2 * K_inv / delta + G) @ x_d)
        term_2 = N * jnp.dot(K @ (x + phi), G @ (x + phi))
        term_3 = -2 * (N + 1) * jnp.dot(x_d_bar, G @ (d + phi))
        term_4 = 2 * jnp.dot(x_d, G @ (x + phi))

        out = -0.5 * (term_1 + term_2 + term_3 + term_4)
        return jnp.nan_to_num(out)

    return log_H
