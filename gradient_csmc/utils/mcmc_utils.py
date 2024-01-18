import jax
import jax.numpy as jnp

from gradient_csmc.utils.pbar import progress_bar_scan


def delta_adaptation_routine(
        key,
        init_xs, init_bs,
        kernel,
        target_acceptance,
        initial_deltas,
        n_steps,
        verbose=False,
        min_delta=1e-12,
        max_delta=1e2,
        min_rate=1e-2,
        window_size=100,
        rate=0.1,
        **_kwargs
):
    T = init_xs.shape[0]

    if verbose:
        decorator = progress_bar_scan(n_steps, show=-1)
    else:
        decorator = lambda x: x

    @decorator
    def body(carry, inp):
        state, deltas, accepted_history, *_ = carry
        xs, bs = state
        i, key_i = inp

        # Run kernel
        next_xs, next_bs, *_ = kernel(key_i, state, deltas)

        accepted = next_bs != bs
        accepted_history = accepted_history.at[:, 1:].set(accepted_history[:, :-1])
        accepted_history = accepted_history.at[:, 0].set(accepted)
        acceptance_rates = jnp.nanmean(accepted_history, 1)

        flag = jnp.logical_or(acceptance_rates < target_acceptance - 0.05,
                              acceptance_rates > target_acceptance + 0.05)
        flag &= i > window_size
        rate_i = jnp.maximum(min_rate, rate / (i + 1) ** 0.5)

        deltas_otherwise = deltas + rate_i * deltas * (
                acceptance_rates - target_acceptance) / target_acceptance

        deltas = jnp.where(flag, deltas_otherwise, deltas)

        deltas = jnp.clip(deltas, min_delta, max_delta)
        carry_out = (next_xs, next_bs), deltas, accepted_history, jnp.mean(acceptance_rates)
        return carry_out, None

    initial_deltas = initial_deltas * jnp.ones(init_xs.shape[0])
    initial_accepted_history = jnp.zeros((T, window_size)) * jnp.nan
    init = (init_xs, init_bs), initial_deltas, initial_accepted_history, jnp.mean(initial_accepted_history)
    inps = jnp.arange(n_steps), jax.random.split(key, n_steps)
    (fin_state, fin_deltas, *_), _ = jax.lax.scan(body, init, inps)
    return fin_state, fin_deltas


def sampling_routine(key,
                     init_xs, init_bs,
                     kernel,
                     n_steps,
                     verbose=False,
                     get_samples=True):
    if verbose:
        decorator = progress_bar_scan(n_steps, show=-1)
    else:
        decorator = lambda x: x

    @decorator
    def body(carry, inp):
        i, key_op = inp
        xs, bs, show = carry

        # Run kernel
        next_xs, next_bs, *_ = kernel(key_op, (xs, bs))
        accepted = next_bs != bs
        carry_out = next_xs, next_bs, jnp.mean(accepted)

        return carry_out, (next_xs, accepted) if get_samples else None

    init = init_xs, init_bs, 0.
    inps = jnp.arange(n_steps), jax.random.split(key, n_steps)
    final_xs, out = jax.lax.scan(body, init, inps)
    if get_samples:
        samples, flags = out
        return samples[::int(get_samples)], flags[::int(get_samples)]
    else:
        return final_xs[:2]
