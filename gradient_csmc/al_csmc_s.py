"""
Implements the Particle-aMALA+ kernel of the paper.
"""
from functools import partial
from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp

from gradient_csmc.csmc import backward_scanning_pass
from gradient_csmc.utils.resamplings import normalize


def kernel(key: PRNGKey, x_star: Array, b_star: Array, Gamma_0: Callable,
           Gamma_t: Union[Callable, tuple[Callable, Any]], ells: Array,
           deltas: Array,
           resampling_func: Callable, ancestor_move_func: Callable, N: int,
           backward: bool = False):
    """
    Particle-aMALA kernel.
    Contrary to the Particle-aMALA kernel, this kernel uses the full gradient of the log-likelihood.

    Parameters
    ----------
    key:
        Random number generator key.
    x_star:
        Reference trajectory to update.
    b_star:
        Indices of the reference trajectory.
    Gamma_0:
        Initial weight function.
    Gamma_t:
        If a tuple, the first element is the function and the second element is the parameters.
        Gamma(None, x, None) returns the log-likelihood at time 0.
    ells:
        Step-size for the random walk.
    deltas:
        Step-size for the Langevin diffusion.
    resampling_func:
        Resampling scheme to use.
    ancestor_move_func:
        Function to move the last ancestor indices.
    N:
        Number of particles to use (N+1, if we include the reference trajectory).
    backward:
        Whether to run the backward sampling kernel.

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
    T, d_x = x_star.shape

    keys = jax.random.split(key, T + 2)
    key_proposals, key_aux, key_backward, keys_resampling = keys[0], keys[1], keys[2], keys[3:]

    # Compute auxiliary variables standard deviations
    aux_std_devs = jnp.sqrt(0.5 * ells)

    # Unpack and modify Gamma_function
    Gamma_t, Gamma_params = Gamma_t if isinstance(Gamma_t, tuple) else (Gamma_t, None)

    @partial(jnp.vectorize, signature="(d),(d)->(),(d),(d)", excluded=(2,))
    def val_grad_Gamma(a, b, p):
        # Wrapper around val_grad_Gamma_ to handle the vectorization of the output.
        out_val, (out_grad_a, out_grad_b) = jax.value_and_grad(Gamma_t, (0, 1))(a, b, p)
        return out_val, out_grad_a, out_grad_b

    val_grad_Gamma_0 = jnp.vectorize(jax.value_and_grad(Gamma_0, 0), signature="(d)->(),(d)")

    ######################################
    #       Gamma tilde function         #
    ######################################

    def Gamma_0_tilde(x_0, delta_0, ell_0, u_0):
        val, grad = val_grad_Gamma_0(x_0)
        mean = x_0 + 0.5 * delta_0 * grad
        return val - jnp.sum((u_0 - mean) ** 2, axis=-1) / ell_0

    def Gamma_1_tilde(x_0, x_1, params_0, params_1):
        delta_0, ell_0, u_0 = params_0
        delta_1, ell_1, u_1, Gamma_params_1 = params_1

        val, grad_0_1, grad_1_1 = val_grad_Gamma(x_0, x_1, Gamma_params_1)
        _, grad_0_0 = val_grad_Gamma_0(x_0)

        mean_0_1 = x_0 + 0.5 * delta_0 * (grad_0_1 + grad_0_0)
        mean_1_1 = x_1 + 0.5 * delta_1 * grad_1_1
        mean_0_0 = x_0 + 0.5 * delta_0 * grad_0_0

        correction_0_1 = -jnp.sum((u_0 - mean_0_1) ** 2, axis=-1) / ell_0
        correction_1_1 = -jnp.sum((u_1 - mean_1_1) ** 2, axis=-1) / ell_1
        correction_0_0 = -jnp.sum((u_0 - mean_0_0) ** 2, axis=-1) / ell_0

        correction = correction_0_1 + correction_1_1 - correction_0_0
        return val + correction

    def Gamma_t_tilde(x_t_m_2, x_t_m_1, x_t, params_t_m_1, params_t):
        delta_t_m_1, ell_t_m_1, u_t_m_1, original_params_t_m_1 = params_t_m_1
        delta_t, ell_t, u_t, original_params_t = params_t

        val, grad_t_m_1_t, grad_t_t = val_grad_Gamma(x_t_m_1, x_t, original_params_t)
        *_, grad_t_m_1_t_m_1 = val_grad_Gamma(x_t_m_2, x_t_m_1, original_params_t_m_1)

        mean_t_m_1_t = x_t_m_1 + 0.5 * delta_t_m_1 * (grad_t_m_1_t + grad_t_m_1_t_m_1)
        mean_t_t = x_t + 0.5 * delta_t * grad_t_t
        mean_t_m_1_t_m_1 = x_t_m_1 + 0.5 * delta_t_m_1 * grad_t_m_1_t_m_1

        correction_t_m_1_t = -jnp.sum((u_t_m_1 - mean_t_m_1_t) ** 2, axis=-1) / ell_t_m_1
        correction_t_t = -jnp.sum((u_t - mean_t_t) ** 2, axis=-1) / ell_t
        correction_t_m_1_t_m_1 = -jnp.sum((u_t_m_1 - mean_t_m_1_t_m_1) ** 2, axis=-1) / ell_t_m_1

        correction = correction_t_m_1_t + correction_t_t - correction_t_m_1_t_m_1
        return val + correction

    #######################################
    #        G_tilde function             #
    #######################################

    def G_0_tilde(x_0, delta_0, ell_0, u_0):
        # Compute N(x_t | u_t, 0.5 * ell_t * I), note how 0.5 / 0.5 = 1
        proposal_log_pdf = -jnp.sum((x_0 - u_0) ** 2, axis=1) / ell_0
        Gamma_val = Gamma_0_tilde(x_0, delta_0, ell_0, u_0)
        return Gamma_val - proposal_log_pdf

    def G_1_tilde(x_0, x_1, params_0, params_1):
        delta_1, ell_1, u_1, Gamma_params_1 = params_1

        proposal_log_pdf = -jnp.sum((x_1 - u_1) ** 2, axis=1) / ell_1
        Gamma_val = Gamma_1_tilde(x_0, x_1, params_0, params_1)
        return Gamma_val - proposal_log_pdf

    def G_t_tilde(x_t_m_2, x_t_m_1, x_t, params_t_m_1, params_t):
        _delta_t, ell_t, u_t, _original_params_t = params_t
        # Compute N(x_t | u_t, 0.5 * ell_t * I), note how 0.5 / 0.5 = 1
        proposal_log_pdf = -jnp.sum((x_t - u_t) ** 2, axis=1) / ell_t
        Gamma_val = Gamma_t_tilde(x_t_m_2, x_t_m_1, x_t, params_t_m_1, params_t)
        return Gamma_val - proposal_log_pdf

    ######################################
    #        Auxiliary proposals         #
    ######################################
    # Sample auxiliary variables: u_t = x_star_t + ∇Gamma_func(xt−1:t) + N(0, ell_t * I)
    # This wrapper is used to compute the log-likelihood and its gradient wrt x_t but not x_t_m_1 (as per argnums=1).

    def full_func(x_traj):
        val_0 = Gamma_0(x_traj[0])
        val_rest = jax.vmap(Gamma_t)(x_traj[:-1], x_traj[1:], Gamma_params)
        return val_0 + jnp.sum(val_rest)

    # Compute gradient
    grad_log_w_star = jax.grad(full_func)(x_star)

    eps_aux = jax.random.normal(key_aux, shape=(T, d_x))
    aux_vars = x_star + 0.5 * deltas[:, None] * grad_log_w_star + aux_std_devs[:, None] * eps_aux

    # Sample proposals: x_t ~ N(u_t, 0.5 * ell_t * I)
    eps_proposals = jax.random.normal(key_proposals, shape=(T, N + 1, d_x))
    xs = aux_vars[:, None, :] + aux_std_devs[:, None, None] * eps_proposals

    # Replace the first particle with the star trajectory
    xs = jax.vmap(lambda a, b, c: a.at[b, :].set(c))(xs, b_star, x_star)

    #################################
    #        Initialisation         #
    #################################
    # Compute initial weights and normalize
    log_w0 = G_0_tilde(xs[0], deltas[0], ells[0], aux_vars[0])
    log_w0 -= jnp.max(log_w0)
    w0 = normalize(log_w0, log_space=False)

    #################################
    #        Forward pass           #
    #################################

    # Do the first step outside of the loop
    A_0 = resampling_func(keys_resampling[0], w0, b_star[0], b_star[1])
    x0 = jnp.take(xs[0], A_0, axis=0)
    Gamma_params_0 = jax.tree_map(lambda z: z[0], Gamma_params)

    Gamma_tilde_params_1 = (deltas[0], ells[0], aux_vars[0]), (deltas[1], ells[1], aux_vars[1], Gamma_params_0)

    log_w1 = G_1_tilde(x0, xs[1], *Gamma_tilde_params_1)

    w1 = normalize(log_w1)

    def body(carry, inp):
        w_t_m_1, x_t_m_1, x_t_m_2 = carry
        params_t_m_1, params_t, x_t, b_star_t_m_1, b_star_t, key_t = inp

        # Conditional resampling
        A_t = resampling_func(key_t, w_t_m_1, b_star_t_m_1, b_star_t)
        x_t_m_2 = jnp.take(x_t_m_2, A_t, axis=0)
        x_t_m_1 = jnp.take(x_t_m_1, A_t, axis=0)

        log_w_t = G_t_tilde(x_t_m_2, x_t_m_1, x_t, params_t_m_1, params_t)
        log_w_t -= jnp.max(log_w_t)
        w_t = normalize(log_w_t)

        # Return next step
        next_carry = w_t, x_t, x_t_m_1
        save = log_w_t, A_t, x_t_m_1

        return next_carry, save

    ## Run forward cSMC
    # Make inputs
    Gamma_tilde_params = deltas[1:], ells[1:], aux_vars[1:], Gamma_params
    Gamma_tilde_params_t_m_1 = jax.tree_map(lambda z: z[:-1], Gamma_tilde_params)
    Gamma_tilde_params_t = jax.tree_map(lambda z: z[1:], Gamma_tilde_params)

    inputs = Gamma_tilde_params_t_m_1, Gamma_tilde_params_t, xs[2:], b_star[1:-1], b_star[2:], keys_resampling[1:]
    # Make init
    init = w1, xs[1], x0

    _, (log_ws, As, xs_m_1_out) = jax.lax.scan(body,
                                               init,
                                               inputs)

    # Insert initial weight and particle
    log_ws = jnp.insert(log_ws, 0, log_w1, axis=0)
    log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)
    As = jnp.insert(As, 0, A_0, axis=0)
    xs_m_1_out = jnp.insert(xs_m_1_out, 0, x0[As[1]], axis=0)

    #################################
    #        Backward pass          #
    #################################
    if backward:
        xs, Bs = backward_sampling_pass(key_backward, Gamma_t_tilde, Gamma_tilde_params, Gamma_1_tilde,
                                        Gamma_tilde_params_1,
                                        b_star[-1], xs, xs_m_1_out, log_ws, ancestor_move_func)
    else:
        xs, Bs = backward_scanning_pass(key_backward, As, b_star[-1], xs, log_ws[-1], ancestor_move_func)
    return xs, Bs, log_ws


def backward_sampling_pass(key, Gamma_tilde_t, Gamma_params_t, Gamma_tilde_1, Gamma_params_1, b_star_T, xs, xs_m_1,
                           log_ws, ancestor_move_func):
    """
    Specialised backward sampling pass for the Particle-aMALA+ kernel.
    This is needed because the model is not Markovian.

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
