import argparse
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax.experimental.host_callback import call

from experiments.stochastic_volatility.kernels import KernelType, get_csmc_kernel
from experiments.stochastic_volatility.model import get_dynamics, get_data, log_pdf
from gradient_csmc.utils.common import force_move, barker_move
from gradient_csmc.utils.resamplings import killing, multinomial

# General config
MIN_DELTA = 1e-5
MAX_DELTA = 1e1
MIN_RATE = 1e-3
ADAPTATION_WINDOW = 100
ADAPTATION_RATE = 0.5
N_LAGS = 1_000

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=128)
parser.add_argument("--D", dest="D", type=int, default=30)
parser.add_argument("--K", dest="K", type=int, default=5)
parser.add_argument("--M", dest="M", type=int, default=4)
parser.add_argument("--tau", dest="tau", type=int, default=10)

parser.add_argument("--n-samples", dest="n_samples", type=int, default=45_000)
parser.add_argument("--adaptation", dest="adaptation", type=int, default=10_000)
parser.add_argument("--burnin", dest="burnin", type=int, default=5_000)
parser.add_argument("--delta-init", dest="delta_init", type=float,
                    default=10 ** (0.5 * (np.log10(MIN_DELTA) + np.log10(MAX_DELTA))))
# default=1.0)
parser.add_argument("--target", dest="target", type=int, default=75)
parser.add_argument("--target-stat", dest='target_stat', type=str, default="mean")

parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.MALA)
parser.add_argument("--style", dest="style", type=str, default="auxiliary")

parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--resampling", dest='resampling', type=str, default="killing")
parser.add_argument("--last-step", dest='last_step', type=str, default="forced")
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

parser.add_argument("--debug", action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument("--verbose", action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)

parser.add_argument("--plot", action='store_true')
parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.set_defaults(plot=False)

args = parser.parse_args()

print(f"""
###############################################
#     STOCHASTIC VOLATILITY EXPERIMENT        #
###############################################
Configuration:
    - T: {args.T}
    - tau: {args.tau}
    - target: {args.target}
    - kernel: {KernelType(args.kernel).name}
    - style: {args.style}
    - D: {args.D}
""")

# BACKEND CONFIG
NOW = time.time()

# PARAMETERS
KEY = jax.random.PRNGKey(args.seed)
EXPERIMENT_KEYS = jax.random.split(KEY, args.K)

# we use the similar parameters as Finke and Thiery.
NU, PHI, RHO = 0., 0.90, 0.25
TAU = args.tau / 100
m0, P0, F, Q, b = get_dynamics(NU, PHI, TAU, RHO, args.D)
TARGET_ALPHA = args.target / 100  # 1 - (1 + args.N) ** (-1 / 2)
if args.target_stat.isnumeric():
    TARGET_STAT = float(args.target_stat) / 100
else:
    TARGET_STAT = args.target_stat

if args.resampling == "killing":
    resampling_fn = killing
elif args.resampling == "multinomial":
    resampling_fn = multinomial
else:
    raise ValueError(f"Unknown resampling {args.resampling}")

if args.last_step == "forced":
    last_step_fn = force_move
elif args.last_step == "barker":
    last_step_fn = barker_move
else:
    raise ValueError(f"Unknown last step {args.last_step}")

kernel_type = KernelType(args.kernel)


def tic_fn(arr):
    time_elapsed = time.time() - NOW
    return np.array(time_elapsed, dtype=arr.dtype), arr


@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    data_key, init_key, adaptation_key, burnin_key, sample_key = jax.random.split(key, 5)

    true_xs, ys, inv_chol_P0, inv_chol_Q = get_data(data_key, NU, PHI, TAU, RHO, args.D, args.T)

    kernel, init, adaptation_loop, experiment_loop = kernel_type.kernel_maker(ys, m0, P0, F, Q, b, N=args.N,
                                                                              resampling_func=resampling_fn,
                                                                              backward=args.backward,
                                                                              ancestor_move_func=last_step_fn,
                                                                              style=args.style)

    adaptation_kernel = kernel
    kernel = jax.jit(kernel)
    adaptation_loop = jax.jit(adaptation_loop, static_argnums=(2, 5, 6), static_argnames=("window_size", "target_stat"))
    experiment_loop = jax.jit(experiment_loop, static_argnums=(2, 3, 4, 5))

    csmc_kernel, csmc_init, *_ = get_csmc_kernel(ys, m0, P0, F, Q, b, N=args.N, resampling_func=resampling_fn,
                                                 backward=True, ancestor_move_func=None, conditional=False)

    # This looks like it's using the true data, but it's not (see, the conditional=False above)
    # We only pass it for the shape of the data.
    init_xs, *_ = csmc_kernel(init_key, csmc_init(true_xs), None)
    init_state = init(init_xs)
    with jax.disable_jit(args.debug):
        adaptation_state, adapted_delta = adaptation_loop(adaptation_key, init_state, adaptation_kernel,
                                                          TARGET_ALPHA,
                                                          args.delta_init, args.adaptation, args.verbose,
                                                          min_delta=MIN_DELTA, max_delta=MAX_DELTA,
                                                          window_size=ADAPTATION_WINDOW,
                                                          rate=ADAPTATION_RATE, min_rate=MIN_RATE,
                                                          target_stat=TARGET_STAT,
                                                          )

    if args.verbose:
        jax.debug.print("Adaptation delta median = {}, min = {}, max = {}", jnp.median(adapted_delta),
                        jnp.min(adapted_delta), jnp.max(adapted_delta))

    delta_kernel = lambda k_, s: kernel(k_, s, adapted_delta)
    burnin_keys = jax.random.split(burnin_key, args.M)
    sample_keys = jax.random.split(sample_key, args.M)

    def get_samples(sample_key_op, init_state_op, all_samples, n_samples):
        return experiment_loop(sample_key_op, init_state_op, delta_kernel, n_samples,
                               args.verbose, all_samples)

    with jax.disable_jit(args.debug):
        burnin_samples, burnin_pct = jax.vmap(get_samples, in_axes=[0, None, None, None], out_axes=0)(burnin_keys,
                                                                                                      adaptation_state,
                                                                                                      False,
                                                                                                      args.burnin)
    output_shape = (jax.ShapeDtypeStruct((), burnin_samples.dtype),
                    jax.ShapeDtypeStruct(burnin_samples.shape, burnin_samples.dtype))
    tic, burnin_samples = call(tic_fn, burnin_samples,
                           result_shape=output_shape)
    burnin_states = jax.vmap(init, in_axes=0)(burnin_samples)


    samples, final_pct = jax.vmap(get_samples, in_axes=[0, 0, None, None], out_axes=1)(sample_keys, burnin_states, True,
                                                                                       args.n_samples)

    final_pct = jnp.mean(final_pct * 1.0, 0)

    output_shape = (jax.ShapeDtypeStruct((), tic.dtype),
                    jax.ShapeDtypeStruct(final_pct.shape, final_pct.dtype))

    toc, final_pct = call(tic_fn, final_pct,
                          result_shape=output_shape)
    final_pct = jnp.reshape(final_pct, (args.M, -1)) * jnp.ones((args.M, args.T))
    energy = log_pdf(samples, ys, m0, inv_chol_P0, F, inv_chol_Q, b)
    if args.M > 1:
        samples_ess = tfp.mcmc.effective_sample_size(samples, filter_beyond_positive_pairs=False,
                                                     cross_chain_dims=1) / args.M
    else:
        samples_ess = tfp.mcmc.effective_sample_size(samples[:, 0, ...], filter_beyond_positive_pairs=False)

    samples_acfs = tfp.stats.auto_correlation(samples, axis=0, max_lags=N_LAGS)
    samples_acf = jnp.mean(samples_acfs, 1)

    means_here = jnp.mean(samples, 0)
    std_devs_here = jnp.std(samples, 0)
    time_here = (toc - tic) / args.M
    return means_here, std_devs_here, samples_ess, final_pct, energy, init_xs, true_xs, ys, adapted_delta, time_here, samples_acf


final_pct_all = np.empty((args.K, args.M, args.T))
ess_all = np.empty((args.K, args.T, args.D))
energy_all = np.empty((args.K, args.n_samples, args.M))
adapted_delta_all = np.empty((args.K, args.T))
sampling_time_all = np.empty((args.K,))
means_all = np.empty((args.K, args.M, args.T, args.D))
std_devs_all = np.empty((args.K, args.M, args.T, args.D))
true_xs_all = np.empty((args.K, args.T, args.D))
ys_all = np.empty((args.K, args.T, args.D))
init_xs_all = np.empty((args.K, args.T, args.D))
acf_all = np.empty((args.K, N_LAGS + 1, args.T, args.D))

for k, key_k in enumerate(EXPERIMENT_KEYS):
    print(f"Running experiment {k + 1}/{args.K}")
    means_k, std_k, ess_k, final_pct_k, energy_k, init_xs_k, true_xs_k, ys_k, adapted_delta_k, sample_time_k, samples_iacf_k = one_experiment(
        key_k)

    final_pct_all[k, ...] = final_pct_k
    ess_all[k, :, :] = np.asarray(ess_k)
    energy_all[k, :] = np.asarray(energy_k)
    saved_delta_k = adapted_delta_k * np.ones((args.T,))
    adapted_delta_all[k, :] = np.asarray(saved_delta_k)
    sampling_time_all[k] = sample_time_k
    means_all[k, ...] = means_k
    std_devs_all[k, ...] = std_k
    true_xs_all[k, ...] = true_xs_k
    ys_all[k, ...] = ys_k
    init_xs_all[k, ...] = init_xs_k
    acf_all[k, ...] = samples_iacf_k

    print(f"""
Results:
    - sampling time: {float(sample_time_k):.0f}s
    - final min-max acceptance rate: {np.min(final_pct_k):.2%}, {np.max(final_pct_k):.2%}
    - final min-max delta: {np.min(saved_delta_k):.2E}, {np.max(saved_delta_k):.2E}
    - final min-max ess: {np.min(ess_k):.2f}, {np.max(ess_k):.2f} 
    - final argmin-argmax ess: {np.argmin(ess_k)}, {np.argmax(ess_k)}
    - final min-max energy: {np.min(energy_k):.2E}, {np.max(energy_k):.2E}
""")
    print()

    if args.plot:
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(final_pct_k.T, alpha=0.5, label="Acceptance rate")
        plt.legend()
        ax.twinx().semilogy(saved_delta_k.T, alpha=0.5, label="Delta")
        plt.legend()
        plt.show()

        for component in range(args.D):
            plt.figure(figsize=(15, 5))
            for m in range(args.M):
                plt.plot(means_k[m, :, component], alpha=0.5, label="Mean")
                plt.fill_between(np.arange(args.T),
                                 means_k[m, :, component] - 2 * std_k[m, :, component],
                                 means_k[m, :, component] + 2 * std_k[m, :, component],
                                 alpha=0.3)
            plt.plot(true_xs_k[:, component], label="True")
            plt.plot(init_xs_k[:, component], label="Init")
            plt.legend()
            plt.show()

if not os.path.exists("results"):
    os.mkdir("results")

file_name = "results/kernel={},T={},D={},tau={},N={},style={},target={:.2f}.npz"
file_name = file_name.format(kernel_type.name, args.T, args.D, args.tau, args.N, args.style, TARGET_ALPHA)

np.savez_compressed(file_name, ess=ess_all, final_pct=final_pct_all, delta=adapted_delta_all,
                    sampling_time=sampling_time_all, energy=energy_all, means=means_all, std_devs=std_devs_all,
                    true_xs=true_xs_all, ys=ys_all, init_xs=init_xs_all, iacf_all=acf_all)
