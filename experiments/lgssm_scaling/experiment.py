import argparse
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from experiments.lgssm_scaling.kernels import KernelType
from experiments.lgssm_scaling.model import get_data
from gradient_csmc.utils.common import force_move, barker_move
from gradient_csmc.utils.kalman import sampling, filtering
from gradient_csmc.utils.resamplings import killing, multinomial

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_platform_name", "cpu")

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=1_000)
parser.add_argument("--D", dest="D", type=int, default=10)
parser.add_argument("--K", dest="K", type=int, default=50)
parser.add_argument("--M", dest="M", type=int, default=10_000)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)

parser.add_argument("--delta", dest="delta", type=float,
                    default=1.)
parser.add_argument("--delta-scale", dest="delta_scale", type=float, default=1 / 3)
parser.add_argument("--delta-arg", dest="delta_arg", type=str, default="na")
parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.CSMC)
parser.add_argument("--style", dest="style", type=str, default="bootstrap")

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
    - kernel: {KernelType(args.kernel).name}
    - style: {args.style}
    - D: {args.D}
""")

# BACKEND CONFIG
NOW = time.time()

# PARAMETERS
KEY = jax.random.PRNGKey(args.seed)
EXPERIMENT_KEYS = jax.random.split(KEY, args.K)

if args.delta_arg == "D":
    DELTA = args.delta / args.D ** args.delta_scale
elif args.delta_arg == "T":
    DELTA = args.delta / args.T ** args.delta_scale
elif args.delta_arg == "DT" or args.delta_arg == "TD":
    DELTA = args.delta / (args.D * args.T) ** args.delta_scale
else:
    DELTA = args.delta

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
DELTA = kernel_type.shape_delta(DELTA, args.T)
SIGMA = 10 ** (args.log_var / 2)


def tic_fn(arr):
    time_elapsed = time.time() - NOW
    return np.array(time_elapsed, dtype=arr.dtype), arr


@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    data_key, init_key, sample_key = jax.random.split(key, 3)

    true_xs, ys = get_data(data_key, SIGMA, args.D, args.T)

    kernel_, init, = kernel_type.kernel_maker(ys, SIGMA, N=args.N,
                                              resampling_func=resampling_fn,
                                              backward=args.backward,
                                              ancestor_move_func=last_step_fn,
                                              style=args.style)

    kernel_ = jax.jit(kernel_)

    # We initialise at stationarity to compute ESJD for fully independent samples.
    m0 = jnp.zeros((args.D,))
    P0 = H = R = jnp.eye(args.D)
    Hs = jnp.repeat(H[None, ...], args.T, axis=0)
    Rs = jnp.repeat(R[None, ...], args.T, axis=0)
    cs = jnp.zeros((args.T, args.D))
    Fs, Qs, bs = Hs[1:], Rs[1:], cs[1:]

    P0 = SIGMA ** 2 * P0
    Qs = SIGMA ** 2 * Qs
    fms, fPs, _ = filtering.filtering(ys, m0, P0, Fs, Qs, bs, Hs, Rs, cs)
    init_xs = sampling.sampling(init_key, fms, fPs, Fs, Qs, bs, args.M)

    init_states = jax.vmap(init)(init_xs)

    sample_keys = jax.random.split(sample_key, args.M)

    def esjd(k_, state):
        xs, *_ = state
        next_xs, *_ = kernel_(k_, state, DELTA)
        return jnp.sum((xs - next_xs) ** 2, -1)

    with jax.disable_jit(args.debug):
        esjd_vals = jax.vmap(esjd)(sample_keys, init_states)

    return esjd_vals.mean(0)


if args.plot:
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

esjd_all = np.empty((args.K, args.T))
for k, key_k in enumerate(tqdm.tqdm(EXPERIMENT_KEYS, desc="Experiment: ")):
    esjd_k = one_experiment(key_k)

    esjd_all[k] = esjd_k

    if args.plot:
        ax.plot(esjd_k, alpha=0.5, label="ESJD")

if args.plot:
    plt.show()

if not os.path.exists("results"):
    os.mkdir("results")
file_name = "results/kernel={},T={},D={},N={},style={},scaling={},scaletype={},log_var={}.csv"
file_name = file_name.format(kernel_type.name, args.T, args.D, args.N, args.style, args.delta_scale, args.delta_arg,
                             args.log_var)

np.savetxt(file_name, esjd_all)
