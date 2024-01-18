"""
Implements the random walk cSMC kernel from Finke and Thiery (2023).
"""
from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp

from gradient_csmc.utils.common import barker_move
from gradient_csmc.utils.resamplings import normalize


def kernel(key: PRNGKey, x_star: Array, b_star: Array, M_0: tuple[Callable, Callable], Gamma_0: Callable,
           M_t: Union[tuple[Callable, Callable], tuple[Callable, Callable, Any]],
           Gamma_t: Union[Callable, tuple[Callable, Any]],
           resampling_func: Callable, ancestor_move_func: Callable, N: int, backward: bool = False,
           conditional: bool = True):
    """
    Generic cSMC kernel.

    Parameters
    ----------
    key:
        Random number generator key.
    x_star:
        Reference trajectory to update.
    b_star:
        Indices of the reference trajectory.
    M_0:
        Sampler for the initial distribution.
        The first element is the sampling function, taking two arguments (key and number of particle)
        and the second element is the logpdf taking a single particle as argument.
    Gamma_0:
        Initial weight function.
    M_t:
        Sampler for the proposal distribution at time t.
        The first element is the sampling function, taking three arguments (key, particles at time t-1, and parameters)
        and the second element is the logpdf taking the particles at time t-1, at time t, and the parameters as arguments.
    Gamma_t:
        If a tuple, the first element is the function and the second element is the parameters.
    resampling_func:
        Resampling scheme to use.
    ancestor_move_func:
        Function to move the last ancestor indices.
    N:
        Number of particles to use (N+1, if we include the reference trajectory).
    backward:
        Whether to run the backward sampling kernel.
    conditional:
        Whether to do conditional SMC or just SMC.

    Returns
    -------

    xs:
        Particles.
    bs:
        Indices of the ancestors.
    """

    As, Gamma_params, Gamma_t, key_backward, log_ws, xs = forward_pass(key, x_star, b_star, M_0, Gamma_0, M_t, Gamma_t,
                                                                       resampling_func, N, conditional)

    #################################
    #        Backward pass          #
    #################################
    if backward:
        xs, Bs = backward_sampling_pass(key_backward, Gamma_t, Gamma_params, b_star[-1], xs, log_ws,
                                        ancestor_move_func, conditional)
    else:
        xs, Bs = backward_scanning_pass(key_backward, As, b_star[-1], xs, log_ws[-1], ancestor_move_func)
    return xs, Bs, log_ws


def forward_pass(key: PRNGKey, x_star: Array, b_star: Array, M_0: tuple[Callable, Callable], Gamma_0: Callable,
                 M_t: Union[tuple[Callable, Callable], tuple[Callable, Callable, Any]],
                 Gamma_t: Union[Callable, tuple[Callable, Any]],
                 resampling_func: Callable, N: int, conditional=True):
    """
        Forward pass of the cSMC kernel.

        Parameters
        ----------
        key:
            Random number generator key.
        x_star:
            Reference trajectory to update.
        b_star:
            Indices of the reference trajectory.
        M_0:
            Sampler for the initial distribution.
            The first element is the sampling function, taking two arguments (key and number of particle)
            and the second element is the logpdf taking a single particle as argument.
        Gamma_0:
            Initial weight function.
        M_t:
            Sampler for the proposal distribution at time t.
            The first element is the sampling function, taking three arguments (key, particles at time t-1, and parameters)
            and the second element is the logpdf taking the particles at time t-1, at time t, and the parameters as arguments.
        Gamma_t:
            If a tuple, the first element is the function and the second element is the parameters.
            Gamma(None, x, None) returns the log-likelihood at time 0.
        resampling_func:
            Resampling scheme to use.
        N:
            Number of particles to use (N+1, if we include the reference trajectory).
        conditional:
            Whether to do conditional SMC or just SMC.

        Returns
        -------

        """
    ###############################
    #        HOUSEKEEPING         #
    ###############################
    T, _d_x = x_star.shape
    key_init, key_loop, key_backward = jax.random.split(key, 3)
    # Unpack Gamma_function
    Gamma_t, Gamma_params = Gamma_t if isinstance(Gamma_t, tuple) else (Gamma_t, None)
    M_t_rvs, M_t_logpdf, prop_params = M_t if len(M_t) == 3 else (*M_t, None)
    M_0_rvs, M_0_logpdf = M_0
    #################################
    #        Initialisation         #
    #################################
    x0 = M_0_rvs(key_init, N + 1)
    if conditional:
        x0 = x0.at[b_star[0]].set(x_star[0])

    # Compute initial weights and normalize
    log_w0 = Gamma_0(x0) - M_0_logpdf(x0)
    log_w0 = normalize(log_w0, log_space=True)
    w0 = jnp.exp(log_w0)

    #################################
    #        Forward pass           #
    #################################
    def body(carry, inp):
        w_t_m_1, x_t_m_1 = carry
        M_t_params, Gamma_params_t, b_star_t_m_1, b_star_t, key_t, x_star_t = inp

        key_proposal_t, key_resampling_t = jax.random.split(key_t, 2)
        # Conditional resampling
        A_t = resampling_func(key_resampling_t, w_t_m_1, b_star_t_m_1, b_star_t, conditional)
        x_t_m_1 = jnp.take(x_t_m_1, A_t, axis=0)

        # Sample proposal
        x_t = M_t_rvs(key_proposal_t, x_t_m_1, M_t_params)
        if conditional:
            x_t = x_t.at[b_star_t].set(x_star_t)

        log_w_t = Gamma_t(x_t_m_1, x_t, Gamma_params_t) - M_t_logpdf(x_t_m_1, x_t, M_t_params)
        log_w_t = normalize(log_w_t, log_space=True)
        w_t = jnp.exp(log_w_t)

        # Return next step
        next_carry = w_t, x_t
        save = log_w_t, A_t, x_t

        return next_carry, save

    keys_loop = jax.random.split(key_loop, T - 1)
    # Run forward cSMC
    inputs = prop_params, Gamma_params, b_star[:-1], b_star[1:], keys_loop, x_star[1:]
    _, (log_ws, As, xs) = jax.lax.scan(body,
                                       (w0, x0),
                                       inputs)
    # Insert initial weight and particle
    log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)
    xs = jnp.insert(xs, 0, x0, axis=0)
    return As, Gamma_params, Gamma_t, key_backward, log_ws, xs


def backward_sampling_pass(key, Gamma_func, Gamma_params, b_star_T, xs, log_ws, ancestor_move_func,
                           conditional: bool = True):
    """
    Backward sampling pass for the cSMC kernel.

    Parameters
    ----------
    key:
        Random number generator key.
    Gamma_func:
        Weight increments function.
    Gamma_params:
        Parameters for the Gamma function.
    b_star_T:
        Index of the last ancestor.
    xs:
        Array of particles.
    log_ws:
        Array of log-weights for the filtering solution.
    ancestor_move_func:
        Function to move the last ancestor indices.
    conditional:
        Whether to do conditional SMC or just SMC.

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
    if not conditional:
        B_T, _ = barker_move(keys[-1], normalize(log_ws[-1]), None)
    else:
        B_T, _ = ancestor_move_func(keys[-1], normalize(log_ws[-1]), b_star_T)
    x_T = xs[-1, B_T]

    def body(x_t, inp):
        op_key, xs_t_m_1, log_w_t_m_1, Gamma_params_t = inp
        Gamma_log_w = Gamma_func(xs_t_m_1, x_t, Gamma_params_t)
        Gamma_log_w -= jnp.max(Gamma_log_w)
        log_w = Gamma_log_w + log_w_t_m_1
        w = normalize(log_w)
        B_t_m_1 = jax.random.choice(op_key, w.shape[0], p=w, shape=())
        x_t_m_1 = xs_t_m_1[B_t_m_1]
        return x_t_m_1, (x_t_m_1, B_t_m_1)

    # Reverse arrays, ideally, should use jax.lax.scan(reverse=True) but it is simpler this way due to insertions.
    # xs[-2::-1] is the reversed list of xs[:-1], I know, not readable... Same for log_ws.
    Gamma_params = jax.tree_map(lambda x: x[::-1], Gamma_params)
    inps = keys[:-1], xs[-2::-1], log_ws[-2::-1], Gamma_params

    # Run backward pass
    _, (xs, Bs) = jax.lax.scan(body, x_T, inps)

    # Insert last ancestor and particle
    xs = jnp.insert(xs, 0, x_T, axis=0)
    Bs = jnp.insert(Bs, 0, B_T, axis=0)

    return xs[::-1], Bs[::-1]


def backward_scanning_pass(key, As, b_star_T, xs, log_w_T, ancestor_move_func):
    """
    Backward scanning pass for the cSMC kernel.

    Parameters
    ----------
    key:
        Random number generator key.
    As:
        Array of indices of the ancestors.
    b_star_T:
        Index of the last ancestor.
    xs:
        Array of particles.
    log_w_T:
        Log-weight of the last ancestor.
    ancestor_move_func:
        Function to move the last ancestor indices.

    Returns
    -------
    xs:
        Array of particles.
    Bs:
        Array of indices of the star trajectory.
    """

    ###############################
    #        BACKWARD PASS        #
    ###############################
    # Select last ancestor
    B_T, _ = ancestor_move_func(key, normalize(log_w_T), b_star_T)
    x_T = xs[-1, B_T]

    def body(B_t, inp):
        xs_t_m_1, A_t = inp
        B_t_m_1 = A_t[B_t]
        x_t_m_1 = xs_t_m_1[B_t_m_1]
        return B_t_m_1, (x_t_m_1, B_t_m_1)

    # xs[-2::-1] is the reversed list of xs[:-1], I know, not readable...
    _, (xs, Bs) = jax.lax.scan(body, B_T, (xs[-2::-1], As[::-1]))
    xs = jnp.insert(xs, 0, x_T, axis=0)
    Bs = jnp.insert(Bs, 0, B_T, axis=0)
    return xs[::-1], Bs[::-1]
