"""
Credit to Jeremie Coullon for this code, see https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/
"""
from jax import lax
from jax.experimental import host_callback
from tqdm.auto import tqdm


def progress_bar_scan(num_samples, message=None, show=None):
    "Progress bar for a JAX scan"
    if message is None:
        message = f"Running for {num_samples:,} iterations"
    tqdm_bars = {}

    if num_samples > 1_000:
        print_rate = int(num_samples / 1_000)
    else:
        print_rate = 1  # if you run the sampler for less than 20 iterations
    remainder = num_samples % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(num_samples))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(args, transform):
        tqdm_bars[0].update(args[0])
        if args[1] is not None:
            tqdm_bars[0].set_description(message + f", carry: {args[1]},", refresh=True)

    def _update_progress_bar(iter_num, carry_show=None):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != num_samples - remainder),
            lambda _: host_callback.id_tap(_update_tqdm, (print_rate, carry_show), result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm by `remainder`
            iter_num == num_samples - remainder,
            lambda _: host_callback.id_tap(_update_tqdm, (remainder, carry_show), result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return lax.cond(
            iter_num == num_samples - 1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            if show is not None:
                carry_show = carry[show]
            else:
                carry_show = None
            _update_progress_bar(iter_num, carry_show)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan
