"""
Implementation of the Particle-aGrad+ kernel.
"""
from functools import partial
from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular

from gradient_csmc.atp_csmc_f import Mt_rvs, Mt_logpdf, M0_rvs, M0_logpdf, make_proposal
from gradient_csmc.csmc import backward_scanning_pass
from gradient_csmc.utils.math import mvn_logpdf
from gradient_csmc.utils.resamplings import normalize


def get_kernel(mu0: Array, P0: Array, r0: Callable,
               mut: Union[Callable, tuple[Callable, Any]], Qs,
               rt: Union[Callable, tuple[Callable, Any]],
               resampling_func: Callable, ancestor_move_func: Callable, N: int,
               backward: bool = False):
    """
    Constructor for the Particle-aGrad kernel.

    Parameters
    ----------
    mu0:
        Initial distribution mean vector.
    P0:
        Initial distribution covariance matrix.
    r0:
        Initial distribution log-density.
    mut:
        Transition function. Either a callable or a tuple of a callable and its parameters.
    Qs:
        Array of covariance matrices. The first element is the covariance at time 0: p(x0) = N(m0, Qs[0]).
        The rest is the covariance matrix of transition functions: p(xt | xt-1) = N(xt | µ(xt-1), Qs[t]).
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

    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Qs = jnp.linalg.cholesky(Qs)

    inv_chol_P0 = solve_triangular(chol_P0, jnp.eye(P0.shape[-1]), lower=True)
    inv_chol_Qs = jax.vmap(lambda z: solve_triangular(z, jnp.eye(Qs.shape[-1]), lower=True))(chol_Qs)

    Qs_ = jnp.insert(Qs, 0, P0, axis=0)
    get_proposal_params = make_proposal(Qs_)

    mut, mut_params = mut if isinstance(mut, tuple) else (mut, None)
    rt, rt_params = rt if isinstance(rt, tuple) else (rt, None)

    val_and_grad_r0 = jnp.vectorize(jax.value_and_grad(r0), signature='(d)->(),(d)')

    @partial(jnp.vectorize, signature="(d),(d)->(),(d),(d)", excluded=(2,))
    def val_and_grad_rt(a, b, p):
        # Wrapper around val_grad_Gamma_ to handle the vectorization of the output.
        out_val, (out_grad_a, out_grad_b) = jax.value_and_grad(rt, (0, 1))(a, b, p)
        return out_val, out_grad_a, out_grad_b

    def kernel(key: PRNGKey, x_star: Array, b_star: Array, ells: Array, deltas: Array):
        """
        Kernel for the Particle-aGrad+ kernel with pre-computed parameters.

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
        T, d_x = x_star.shape
        keys = jax.random.split(key, T + 2)
        key_init, key_aux, key_backward, keys_rest = keys[0], keys[1], keys[2], keys[3:]

        # Compute auxiliary variables standard deviations
        aux_std_devs = jnp.sqrt(0.5 * ells)

        ######################################
        #        Auxiliary proposals         #
        ######################################
        # Sample auxiliary variables: u_t = x_star_t + ∇Gamma_func(xt−1:t) + N(0, ell_t * I)

        def full_func(x_traj):
            val_0 = r0(x_traj[0])
            val_rest = jax.vmap(rt)(x_traj[:-1], x_traj[1:], rt_params)
            return val_0 + jnp.sum(val_rest)

        # Compute gradient
        grad_log_w_star = jax.grad(full_func)(x_star)

        eps_aux = jax.random.normal(key_aux, shape=(T, d_x))
        aux_vars = x_star + 0.5 * deltas[:, None] * grad_log_w_star + aux_std_devs[:, None] * eps_aux

        #################################
        #      Proposal and functions   #
        #################################
        Ks_1, Ks_2, chols_prop, chols_inv_prop = get_proposal_params(ells)

        spec_M0_rvs = partial(M0_rvs, params=(aux_vars[0], mu0, Ks_1[0], Ks_2[0], chols_prop[0]))
        spec_M0_logpdf = partial(M0_logpdf, params=(aux_vars[0], mu0, Ks_1[0], Ks_2[0], chols_inv_prop[0]))
        spec_Mt_rvs = partial(Mt_rvs, mut=mut)
        spec_Mt_logpdf = partial(Mt_logpdf, mut=mut)

        Mt_params = (aux_vars[1:], Ks_1[1:], Ks_2[1:], chols_prop[1:], chols_inv_prop[1:]), mut_params

        ######################################
        #       Gamma tilde function         #
        ######################################

        def Gamma_0_tilde(x, delta, ell, u):
            out, grad = val_and_grad_r0(x)
            mean = x + 0.5 * delta * grad
            out += mvn_logpdf(x, mu0, None, chol_inv=inv_chol_P0, constant=False)
            correction = -jnp.sum((u - mean) ** 2, axis=-1) / ell
            return out + correction

        def Gamma_1_tilde(x_0, x_1, params_0, params_1):
            delta_0, ell_0, u_0 = params_0
            delta_1, ell_1, u_1, r_params_1, mu_params_1 = params_1
            out, grad_0_1, grad_1_1 = val_and_grad_rt(x_0, x_1, r_params_1)
            _, grad_0_0 = val_and_grad_r0(x_0)

            out += mvn_logpdf(x_1, mut(x_0, mu_params_1), None, chol_inv=inv_chol_Qs[0], constant=False)

            mean_0_1 = x_0 + 0.5 * delta_0 * (grad_0_1 + grad_0_0)
            mean_1_1 = x_1 + 0.5 * delta_1 * grad_1_1
            mean_0_0 = x_0 + 0.5 * delta_0 * grad_0_0

            correction_0_1 = -jnp.sum((u_0 - mean_0_1) ** 2, axis=-1) / ell_0
            correction_1_1 = -jnp.sum((u_1 - mean_1_1) ** 2, axis=-1) / ell_1
            correction_0_0 = -jnp.sum((u_0 - mean_0_0) ** 2, axis=-1) / ell_0

            correction = correction_0_1 + correction_1_1 - correction_0_0
            return out + correction

        def Gamma_t_tilde(x_t_m_2, x_t_m_1, x_t, params_t_m_1, params_t):
            delta_t_m_1, ell_t_m_1, u_t_m_1, r_params_t_m_1, *_ = params_t_m_1
            delta_t, ell_t, u_t, r_params_t, mu_params_t, inv_chol_Qt = params_t

            out, grad_t_m_1_t, grad_t_t = val_and_grad_rt(x_t_m_1, x_t, r_params_t)
            *_, grad_t_m_1_t_m_1 = val_and_grad_rt(x_t_m_2, x_t_m_1, r_params_t_m_1)

            out += mvn_logpdf(x_t, mut(x_t_m_1, mu_params_t), None, chol_inv=inv_chol_Qt, constant=False)

            mean_t_m_1_t = x_t_m_1 + 0.5 * delta_t_m_1 * (grad_t_m_1_t + grad_t_m_1_t_m_1)
            mean_t_t = x_t + 0.5 * delta_t * grad_t_t
            mean_t_m_1_t_m_1 = x_t_m_1 + 0.5 * delta_t_m_1 * grad_t_m_1_t_m_1

            correction_t_m_1_t = -jnp.sum((u_t_m_1 - mean_t_m_1_t) ** 2, axis=-1) / ell_t_m_1
            correction_t_t = -jnp.sum((u_t - mean_t_t) ** 2, axis=-1) / ell_t
            correction_t_m_1_t_m_1 = -jnp.sum((u_t_m_1 - mean_t_m_1_t_m_1) ** 2, axis=-1) / ell_t_m_1

            correction = correction_t_m_1_t + correction_t_t - correction_t_m_1_t_m_1
            return out + correction

        #######################################
        #        G_tilde function             #
        #######################################

        def G_0_tilde(x_0, delta_0, ell_0, u_0):
            # Compute N(x_t | u_t, 0.5 * ell_t * I), note how 0.5 / 0.5 = 1
            Gamma_val = Gamma_0_tilde(x_0, delta_0, ell_0, u_0)
            return Gamma_val

        def G_1_tilde(x_0, x_1, params_0, params_1):
            delta_1, ell_1, u_1, *_ = params_1

            Gamma_val = Gamma_1_tilde(x_0, x_1, params_0, params_1)
            return Gamma_val

        def G_t_tilde(x_t_m_2, x_t_m_1, x_t, params_t_m_1, params_t):
            _delta_t, ell_t, u_t, *_ = params_t
            # Compute N(x_t | u_t, 0.5 * ell_t * I), note how 0.5 / 0.5 = 1
            Gamma_val = Gamma_t_tilde(x_t_m_2, x_t_m_1, x_t, params_t_m_1, params_t)
            return Gamma_val

        #################################
        #        Initialisation         #
        #################################
        # Compute initial weights and normalize
        key_init_0, key_init_1 = jax.random.split(key_init, 2)
        x0 = spec_M0_rvs(key_init_0, N + 1)
        log_w0 = G_0_tilde(x0, deltas[0], ells[0], aux_vars[0])
        log_w0 -= spec_M0_logpdf(x0)
        log_w0 -= jnp.max(log_w0)
        w0 = normalize(log_w0, log_space=False)

        #################################
        #        Forward pass           #
        #################################

        # Do the first step outside the loop
        A_0 = resampling_func(keys_rest[0], w0, b_star[0], b_star[1])
        x0_resampled = jnp.take(x0, A_0, axis=0)

        Mt_params_0 = jax.tree_map(lambda z: z[0], Mt_params)
        x1 = spec_Mt_rvs(key_init_1, x0_resampled, Mt_params_0)

        mut_params_0 = jax.tree_map(lambda z: z[0], mut_params)
        rt_params_0 = jax.tree_map(lambda z: z[0], rt_params)
        Gamma_tilde_params_1 = ((deltas[0], ells[0], aux_vars[0]),
                                (deltas[1], ells[1], aux_vars[1], rt_params_0, mut_params_0))

        log_w1 = G_1_tilde(x0_resampled, x1, *Gamma_tilde_params_1)
        log_w1 -= spec_Mt_logpdf(x0_resampled, x1, Mt_params_0)
        w1 = normalize(log_w1)

        def body(carry, inp):
            w_t_m_1, x_t_m_1, x_t_m_2 = carry
            Mt_params_t, params_t_m_1, params_t, b_star_t_m_1, b_star_t, key_t = inp

            key_resampling, key_proposal = jax.random.split(key_t, 2)

            # Conditional resampling
            A_t = resampling_func(key_resampling, w_t_m_1, b_star_t_m_1, b_star_t)
            x_t_m_2 = jnp.take(x_t_m_2, A_t, axis=0)
            x_t_m_1 = jnp.take(x_t_m_1, A_t, axis=0)

            # Proposal
            x_t = spec_Mt_rvs(key_proposal, x_t_m_1, Mt_params_t)
            log_w_t = G_t_tilde(x_t_m_2, x_t_m_1, x_t, params_t_m_1, params_t)
            log_w_t -= spec_Mt_logpdf(x_t_m_1, x_t, Mt_params_t)
            log_w_t -= jnp.max(log_w_t)
            w_t = normalize(log_w_t)

            # Return next step
            next_carry = w_t, x_t, x_t_m_1
            save = log_w_t, A_t, x_t_m_1, x_t

            return next_carry, save

        ## Run forward cSMC
        # Make inputs
        Gamma_tilde_params = deltas[1:], ells[1:], aux_vars[1:], rt_params, mut_params, inv_chol_Qs
        Gamma_tilde_params_t_m_1 = jax.tree_map(lambda z: z[:-1], Gamma_tilde_params)
        Gamma_tilde_params_t = jax.tree_map(lambda z: z[1:], Gamma_tilde_params)
        inputs = (jax.tree_map(lambda z: z[1:], Mt_params), Gamma_tilde_params_t_m_1, Gamma_tilde_params_t,
                  b_star[1:-1], b_star[2:], keys_rest[1:])
        # Make init
        init = w1, x1, x0_resampled

        _, (log_ws, As, xs_m_1_out, xs_out) = jax.lax.scan(body,
                                                           init,
                                                           inputs)

        xs_out = jnp.insert(xs_out, 0, x1, axis=0)
        xs_out = jnp.insert(xs_out, 0, x0, axis=0)

        # Insert initial weight and particle
        log_ws = jnp.insert(log_ws, 0, log_w1, axis=0)
        log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)
        As = jnp.insert(As, 0, A_0, axis=0)
        xs_m_1_out = jnp.insert(xs_m_1_out, 0, x0_resampled[As[1]], axis=0)

        #################################
        #        Backward pass          #
        #################################
        if backward:
            xs, Bs = backward_sampling_pass(key_backward, Gamma_t_tilde, Gamma_tilde_params, Gamma_1_tilde,
                                            Gamma_tilde_params_1,
                                            b_star[-1], xs_out, xs_m_1_out, log_ws, ancestor_move_func)
        else:
            xs, Bs = backward_scanning_pass(key_backward, As, b_star[-1], xs_out, log_ws[-1], ancestor_move_func)

        return xs, Bs

    return kernel


def backward_sampling_pass(key, Gamma_tilde_t, Gamma_params_t, Gamma_tilde_1, Gamma_params_1, b_star_T, xs, xs_m_1,
                           log_ws, ancestor_move_func):
    """
    Specialised backward sampling pass for the Particle-aGrad+ kernel.
    This is needed because the model is non-Markovian.

    Parameters
    ----------
    key:
        Random number generator key.
    Gamma_tilde_t:
        Weight increments function.
    Gamma_params_t:
        Parameters for the Gamma_tilde function.
    Gamma_tilde_1:
        Weight increments function for the first time step.
    Gamma_params_1:
        Parameters for the Gamma_tilde function for the first time step.
    b_star_T:
        Index of the last ancestor.
    xs:
        Array of particles.
    xs_m_1:
        Ancestors of the particles.
    log_ws:
        Array of log-weights for the filtering solution.
    ancestor_move_func:
        Function to move the last ancestor indices.

    Returns
    -------
    xs:
        Array of particles.
    Bs:
        Array of indices of the last ancestor.
    """
    ###############################
    #        HOUSEKEEPING         #
    ###############################

    T = xs.shape[0]
    keys = jax.random.split(key, T)

    ###############################
    #        BACKWARD PASS        #
    ###############################
    # Select last ancestor
    B_T, _ = ancestor_move_func(keys[-1], normalize(log_ws[-1]), b_star_T)
    x_T = xs[-1, B_T]

    #################################
    #        Do T-1 explicitly      #
    #################################
    # Compute log-weights

    Gamma_params_T = jax.tree_map(lambda z: z[-1], Gamma_params_t)
    Gamma_params_T_m_1 = jax.tree_map(lambda z: z[-2], Gamma_params_t)
    Gamma_log_w_T_m_1 = Gamma_tilde_t(xs_m_1[-2], xs[-2], x_T, Gamma_params_T_m_1, Gamma_params_T)
    log_w_T_m_1 = Gamma_log_w_T_m_1 + log_ws[-2]
    w_T_m_1 = normalize(log_w_T_m_1)
    B_T_m_1 = jax.random.choice(keys[-2], w_T_m_1.shape[0], p=w_T_m_1, shape=())
    x_T_m_1 = xs[-2, B_T_m_1]

    def body(carry, inp):
        t, x_t_p_1, x_t_p_2 = carry
        op_key, xs_t_m_1, xs_t, log_w_t, params_t, params_t_p_1, params_t_p_2 = inp

        Gamma_log_w_t_p_2 = Gamma_tilde_t(xs_t, x_t_p_1, x_t_p_2, params_t_p_1, params_t_p_2)
        Gamma_log_w_t_p_1 = Gamma_tilde_t(xs_t_m_1, xs_t, x_t_p_1, params_t, params_t_p_1)
        w_t = normalize(Gamma_log_w_t_p_2 + Gamma_log_w_t_p_1 + log_w_t)
        B_t = jax.random.choice(op_key, w_t.shape[0], p=w_t, shape=())
        x_t = xs_t[B_t]
        return (t - 1, x_t, x_t_p_1), (x_t, B_t)

    Gamma_params_t_inp = jax.tree_map(lambda z: z[:-2], Gamma_params_t)
    Gamma_params_t_p_1_inp = jax.tree_map(lambda z: z[1:-1], Gamma_params_t)
    Gamma_params_t_p_2_inp = jax.tree_map(lambda z: z[2:], Gamma_params_t)

    inps = (keys[1:-2], xs_m_1[:-2], xs[1:-2],
            log_ws[1:-2], Gamma_params_t_inp, Gamma_params_t_p_1_inp, Gamma_params_t_p_2_inp)

    # Run backward pass
    (t_fin, *_), (xs_out, Bs) = jax.lax.scan(body, (T - 3, x_T_m_1, x_T), inps, reverse=True)

    # append values
    xs_out = jnp.append(xs_out, x_T_m_1[None, :], axis=0)
    xs_out = jnp.append(xs_out, x_T[None, :], axis=0)

    Bs = jnp.append(Bs, B_T_m_1[None], axis=0)
    Bs = jnp.append(Bs, B_T[None], axis=0)

    # Do the final step explicitly
    Gamma_params_2_0 = jax.tree_map(lambda z: z[0], Gamma_params_t)
    Gamma_params_2_1 = jax.tree_map(lambda z: z[1], Gamma_params_t)

    Gamma_log_w_0 = Gamma_tilde_1(xs[0], xs_out[0], *Gamma_params_1)
    Gamma_log_w_1 = Gamma_tilde_t(xs[0], xs_out[0], xs_out[1], Gamma_params_2_0, Gamma_params_2_1)
    log_w_0 = Gamma_log_w_0 + Gamma_log_w_1 + log_ws[0]
    w_0 = normalize(log_w_0)
    B_0 = jax.random.choice(keys[0], w_0.shape[0], p=w_0, shape=())

    # insert initial values
    xs_out = jnp.insert(xs_out, 0, xs[0, B_0], axis=0)
    Bs = jnp.insert(Bs, 0, B_0, axis=0)

    return xs_out, Bs
