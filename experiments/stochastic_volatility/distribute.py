# ARGS PARSING
import argparse
import os
from itertools import product

from experiments.stochastic_volatility.kernels import KernelType

parser = argparse.ArgumentParser()
parser.add_argument("--i", dest="i", type=int, default=0)

TS = (128,)
TAUS = (10, 50, 100, 200,)

KERNELS = (
    KernelType.CSMC,
    KernelType.RW_CSMC,
    KernelType.TP_CSMC, KernelType.TP_CSMC, KernelType.TP_CSMC, KernelType.TP_CSMC, KernelType.TP_CSMC,
    KernelType.MALA_CSMC, KernelType.MALA_CSMC, KernelType.MALA_CSMC, KernelType.MALA_CSMC, KernelType.MALA_CSMC,
    KernelType.ADAPTED_CSMC, KernelType.ADAPTED_CSMC, KernelType.ADAPTED_CSMC,
    KernelType.TP,
    KernelType.MALA, KernelType.MALA,
)

STYLES = (
    'bootstrap',
    'na',
    'filtering', 'marginal', 'twisted',
    'filtering', 'marginal', 'smoothing',
    'filtering', 'marginal', 'twisted',
    'na',
    'auxiliary', 'marginal'
)

TARGET_STATS = (
    "mean",
    "mean",
    "mean", "mean", "mean",
    "mean", "mean", "mean",
    "mean", "mean", "mean",
    "mean",
    "mean", "mean"
)

TARGETS = (75,)

combination = list(product(TS, TAUS, TARGETS, zip(KERNELS, STYLES)))
print(len(combination))
args = parser.parse_args()

T, tau, target, (kernel, style, *_) = combination[args.i]

exec_str = "JAX_PLATFORM_NAME=cpu python3 experiment.py --target {} --tau {} --T {} --kernel {} --target-stat mean --style {} --D 30 --N 31 --verbose"

os.system(exec_str.format(target, tau, T, kernel.value, style))
