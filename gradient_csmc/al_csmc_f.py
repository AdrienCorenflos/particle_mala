"""
Implements the Particle-aMALA kernel of the paper.
"""
from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp

from gradient_csmc.csmc import backward_sampling_pass, backward_scanning_pass
from gradient_csmc.utils.resamplings import normalize


def kernel(key: PRNGKey, x_star: Array, b_star: Array, Gamma_0: Callable,
           Gamma_t: Union[Callable, tuple[Callable, Any]], ells: Array,
           deltas: Array,
           resampling_func: Callable, ancestor_move_func: Callable, N: int,
           backward: bool = False):
    """
    Particle-aMALA kernel.

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
    Gamma_t_grad = jax.grad(Gamma_t, 1)
    Gamma_0_grad = jax.grad(Gamma_0, 0)

    ###########################################
    #       Modified weight functions         #
    ###########################################

    def Gamma_0_tilde(x_0, delta_0, ell_0, u_0):
        val_grad_Gamma = jax.value_and_grad(Gamma_0, 0)
        vec_Gamma_val_grad = jnp.vectorize(val_grad_Gamma, signature='(n)->(),(n)')
        Gamma_val, Gamma_grad = vec_Gamma_val_grad(x_0)

        # Compute N(u_t | x_t + 0.5 * delta_t * Gamma_grad, 0.5 * ell_t * I)
        gradient_log_pdf = -jnp.sum((u_0 - x_0 - 0.5 * delta_0 * Gamma_grad) ** 2, axis=-1) / ell_0
        return Gamma_val + gradient_log_pdf

    def G_0_tilde(x_0, delta_0, ell_0, u_0):
        # Compute N(x_t | u_t, 0.5 * ell_t * I), note how 0.5 / 0.5 = 1
        proposal_log_pdf = -jnp.sum((x_0 - u_0) ** 2, axis=1) / ell_0
        return Gamma_0_tilde(x_0, delta_0, ell_0, u_0) - proposal_log_pdf

    def Gamma_t_tilde(x_t_m_1, x_t, params):
        delta_t, ell_t, u_t, original_params_t = params
        val_grad_Gamma = jax.value_and_grad(Gamma_t, 1)
        vec_Gamma_val_grad = jnp.vectorize(val_grad_Gamma, signature='(n),(n)->(),(n)', excluded=(2,))
        Gamma_val, Gamma_grad = vec_Gamma_val_grad(x_t_m_1, x_t, original_params_t)
        # Gamma_val shape is (N+1,), Gamma_grad shape is (N+1, d_x), u_t shape is (d_x,), x_t shape is (N+1, d_x)

        # Compute N(u_t | x_t + 0.5 * delta_t * Gamma_grad, 0.5 * ell_t * I)
        gradient_log_pdf = -jnp.sum((u_t - x_t - 0.5 * delta_t * Gamma_grad) ** 2, axis=-1) / ell_t
        return Gamma_val + gradient_log_pdf

    def G_t_tilde(x_t_m_1, x_t, params):
        _, ell_t, u_t, original_params_t = params
        # Compute N(x_t | u_t, 0.5 * ell_t * I), note how 0.5 / 0.5 = 1
        proposal_log_pdf = -jnp.sum((x_t - u_t) ** 2, axis=1) / ell_t
        return Gamma_t_tilde(x_t_m_1, x_t, params) - proposal_log_pdf

    ######################################
    #        Auxiliary proposals         #
    ######################################
    # Sample auxiliary variables: u_t = x_star_t + ∇Gamma_func(xt−1:t) + N(0, ell_t * I)
    # This wrapper is used to compute the log-likelihood and its gradient wrt x_t but not x_t_m_1 (as per argnums=1).

    # Compute gradient
    grad_log_w_star_0 = Gamma_0_grad(x_star[0])
    grad_log_w_star = jax.vmap(Gamma_t_grad, [0, 0, 0])(x_star[:-1], x_star[1:], Gamma_params)
    grad_log_w_star = jnp.insert(grad_log_w_star, 0, grad_log_w_star_0, axis=0)

    eps_aux = jax.random.normal(key_aux, shape=(T, d_x))
    aux_vars = x_star + 0.5 * deltas[:, None] * grad_log_w_star + aux_std_devs[:, None] * eps_aux

    # Sample proposals: x_t = u_t + N(0, 0.5 * ell_t * I)
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

    def body(carry, inp):
        w_t_m_1, x_t_m_1 = carry
        Gamma_tilde_params_t, x_t, b_star_t_m_1, b_star_t, key_t = inp

        # Conditional resampling
        A_t = resampling_func(key_t, w_t_m_1, b_star_t_m_1, b_star_t)
        x_t_m_1 = jnp.take(x_t_m_1, A_t, axis=0)

        log_w_t = G_t_tilde(x_t_m_1, x_t, Gamma_tilde_params_t)
        log_w_t -= jnp.max(log_w_t)
        w_t = normalize(log_w_t)

        # Return next step
        next_carry = w_t, x_t
        save = log_w_t, A_t

        return next_carry, save

    # Run forward cSMC
    Gamma_tilde_params = deltas[1:], ells[1:], aux_vars[1:], Gamma_params
    inputs = Gamma_tilde_params, xs[1:], b_star[:-1], b_star[1:], keys_resampling
    _, (log_ws, As) = jax.lax.scan(body,
                                   (w0, xs[0]),
                                   inputs)

    # Insert initial weight and particle
    log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)

    #################################
    #        Backward pass          #
    #################################
    if backward:
        xs, Bs = backward_sampling_pass(key_backward, Gamma_t_tilde, Gamma_tilde_params, b_star[-1], xs, log_ws,
                                        ancestor_move_func)
    else:
        xs, Bs = backward_scanning_pass(key_backward, As, b_star[-1], xs, log_ws[-1], ancestor_move_func)
    return xs, Bs, log_ws
