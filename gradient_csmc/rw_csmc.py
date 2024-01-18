"""
Implements the Particle-RWM kernel from Finke and Thiery (2023).
"""
from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp

from gradient_csmc.csmc import backward_sampling_pass, backward_scanning_pass
from gradient_csmc.utils.resamplings import normalize


def kernel(key: PRNGKey, x_star: Array, b_star: Array, Gamma_0: Callable,
           Gamma_t: Union[Callable, tuple[Callable, Any]], ells: Array,
           resampling_func: Callable, ancestor_move_func: Callable, N: int, backward: bool = False):
    """
    Particle-RWM kernel.
    
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
        Particles.
    bs:
        Indices of the ancestors.
    """
    ###############################
    #        HOUSEKEEPING         #
    ###############################
    T, d_x = x_star.shape
    keys = jax.random.split(key, T + 2)
    key_proposals, key_aux, key_backward, keys_resampling = keys[0], keys[1], keys[2], keys[3:]

    # Compute auxiliary variables standard deviations
    aux_std_devs = jnp.sqrt(0.5 * ells)

    # Unpack Gamma_function
    Gamma_t, Gamma_params = Gamma_t if isinstance(Gamma_t, tuple) else (Gamma_t, None)

    ######################################
    #        Auxiliary proposals         #
    ######################################
    # Sample auxiliary variables: u_t = x_star_t + N(0, ell_t * I)
    aux_vars = x_star + jax.random.normal(key_aux, shape=(T, d_x)) * aux_std_devs[:, None]

    # Sample proposals: x_t = u_t + N(0, 0.5 * ell_t * I)
    eps_xs = jax.random.normal(key_proposals, shape=(T, N + 1, d_x))
    xs = aux_vars[:, None, :] + aux_std_devs[:, None, None] * eps_xs

    # Replace the first particle with the star trajectory
    # xs = xs.at[:, b_star].set(x_star)
    xs = jax.vmap(lambda a, b, c: a.at[b, :].set(c))(xs, b_star, x_star)

    #################################
    #        Initialisation         #
    #################################
    # Compute initial weights and normalize
    log_w0 = Gamma_0(xs[0])
    log_w0 -= jnp.max(log_w0)
    w0 = normalize(log_w0, log_space=False)

    #################################
    #        Forward pass           #
    #################################

    def body(carry, inp):
        w_t_m_1, x_t_m_1 = carry
        Gamma_params_t, x_t, b_star_t_m_1, b_star_t, key_t = inp

        # Conditional resampling
        A_t = resampling_func(key_t, w_t_m_1, b_star_t_m_1, b_star_t)
        x_t_m_1 = jnp.take(x_t_m_1, A_t, axis=0)

        log_w_t = Gamma_t(x_t_m_1, x_t, Gamma_params_t)
        log_w_t = normalize(log_w_t, log_space=True)
        w_t = jnp.exp(log_w_t)
        # Return next step
        next_carry = w_t, x_t
        save = log_w_t, A_t

        return next_carry, save

    # Run forward cSMC
    inputs = Gamma_params, xs[1:], b_star[:-1], b_star[1:], keys_resampling
    _, (log_ws, As) = jax.lax.scan(body,
                                   (w0, xs[0]),
                                   inputs)

    # Insert initial weight and particle
    log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)

    #################################
    #        Backward pass          #
    #################################
    if backward:
        xs, Bs = backward_sampling_pass(key_backward, Gamma_t, Gamma_params, b_star[-1], xs, log_ws,
                                        ancestor_move_func)
    else:
        xs, Bs = backward_scanning_pass(key_backward, As, b_star[-1], xs, log_ws[-1], ancestor_move_func)

    is_any_nan = ~jnp.all(jnp.isfinite(xs))
    xs = jnp.where(is_any_nan, x_star, xs)
    Bs = jnp.where(is_any_nan, b_star, Bs)
    return xs, Bs, log_ws
