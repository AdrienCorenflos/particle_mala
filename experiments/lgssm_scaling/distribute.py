# ARGS PARSING
import argparse
import os
from itertools import product

import numpy as np

from experiments.lgssm_scaling.kernels import KernelType

parser = argparse.ArgumentParser()
parser.add_argument("--i", dest="i", type=int, default=-1)

BIG_T = (25,)
SMALL_T = (10,)
SMALL_D = (10,)
LOG_SIGMA_2 = (0,)

DS = np.logspace(0, 2, 50, dtype=int)
DS = np.unique(DS)

TS = np.logspace(0, 3, 75, dtype=int)
TS = np.unique(TS)

LOG_SIGMAS = [round(x, 2) for x in np.arange(-6, 3.1, 0.1)]

KERNELS = (
    KernelType.CSMC,
    KernelType.RW_CSMC,
    KernelType.TP_CSMC, KernelType.TP_CSMC, KernelType.TP_CSMC, KernelType.TP_CSMC, KernelType.TP_CSMC,
    KernelType.MALA_CSMC, KernelType.MALA_CSMC, KernelType.MALA_CSMC, KernelType.MALA_CSMC, KernelType.MALA_CSMC,
    KernelType.ADAPTED_CSMC, KernelType.ADAPTED_CSMC, KernelType.ADAPTED_CSMC,
    KernelType.TP,
    KernelType.MALA, KernelType.MALA,
    KernelType.RW,
    KernelType.IMH
)

STYLES = (
    'bootstrap',
    'na',
    'filtering', 'marginal', 'filtering', 'marginal', 'twisted',
    'filtering', 'marginal', 'filtering', 'marginal', 'smoothing',
    'filtering', 'marginal', 'twisted',
    'na',
    'auxiliary', 'marginal',
    'auxiliary',
    'na'
)

SCALES = (
    1.,
    1.,
    1., 1., 1 / 3, 1 / 3, 1 / 3,
    1., 1., 1 / 3, 1 / 3, 1 / 3,
    1., 1., 1.,
    1 / 3,
    1 / 3, 1 / 3,
    1.,
    1.
)

DELTA_ARGS = (
    'D',
    'D',
    'D', 'D', 'D', 'D', 'D',
    'D', 'D', 'D', 'D', 'D',
    'D', 'D', 'D',
    'TD',
    'TD', 'TD',
    'TD',
    'TD',
)


combination_1 = list(product(LOG_SIGMA_2, BIG_T, DS, zip(KERNELS, STYLES, SCALES, DELTA_ARGS)))
combination_2 = list(product(LOG_SIGMA_2, TS, SMALL_D, zip(KERNELS, STYLES, SCALES, DELTA_ARGS)))
combination_3 = list(product(LOG_SIGMAS, SMALL_T, SMALL_D, zip(KERNELS, STYLES, SCALES, DELTA_ARGS)))

combination = combination_3
print(len(combination))
args = parser.parse_args()

log_var, T, d, (kernel, style, scale, delta_arg) = combination[args.i]

exec_str = "JAX_PLATFORM_NAME=cpu python3 experiment.py --delta-scale {} --delta-arg {} --T {} --kernel {} --style {} --D {} --N 31 --log-var {} --verbose --M 1000"

os.system(exec_str.format(scale, delta_arg, T, kernel.value, style, d, log_var))
