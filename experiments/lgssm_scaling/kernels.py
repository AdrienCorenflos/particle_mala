from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

import gradient_csmc.al_csmc_f as alf
import gradient_csmc.al_csmc_s as als
import gradient_csmc.atp_csmc_f as atpf
import gradient_csmc.atp_csmc_s as atps
import gradient_csmc.csmc as csmc
import gradient_csmc.l_csmc_f as lf
import gradient_csmc.mala as mala
import gradient_csmc.rw_csmc as rw
import gradient_csmc.t_atp_csmc_f as t_atpf
import gradient_csmc.tp as tp
import gradient_csmc.tp_csmc as tpf
import gradient_csmc.imh as imh
from experiments.lgssm_scaling.model import log_likelihood, log_potential, log_pdf


class KernelType(Enum):
    CSMC = 0
    TP = 1
    TP_CSMC = 2
    MALA_CSMC = 3
    ADAPTED_CSMC = 4
    RW_CSMC = 5
    MALA = 6
    RW = 7
    IMH = 8

    @property
    def kernel_maker(self):
        if self == KernelType.TP:
            return get_tp_kernel
        elif self == KernelType.CSMC:
            return get_csmc_kernel
        elif self == KernelType.TP_CSMC:
            return get_tp_csmc_kernel
        elif self == KernelType.MALA_CSMC:
            return get_mala_csmc_kernel
        elif self == KernelType.ADAPTED_CSMC:
            return partial(get_tp_csmc_kernel, stop_gradient=True)
        elif self == KernelType.RW_CSMC:
            return get_rw_csmc_kernel
        elif self == KernelType.MALA:
            return partial(get_mala_kernel, stop_gradient=False)
        elif self == KernelType.RW:
            return partial(get_mala_kernel, stop_gradient=True)
        elif self == KernelType.IMH:
            return get_imh_kernel
        else:
            raise NotImplementedError

    def shape_delta(self, delta, T):
        if self == KernelType.TP:
            return delta
        elif self == KernelType.CSMC:
            return delta
        elif self == KernelType.TP_CSMC:
            return delta * np.ones((T,))
        elif self == KernelType.MALA_CSMC:
            return delta * np.ones((T,))
        elif self == KernelType.ADAPTED_CSMC:
            return delta * np.ones((T,))
        elif self == KernelType.RW_CSMC:
            return delta * np.ones((T,))
        elif self == KernelType.RW:
            return delta
        elif self == KernelType.IMH:
            return delta
        elif self == KernelType.MALA:
            return delta


#######################
# Kernel constructors #
#######################


def get_tp_kernel(ys, sigma, N=1, **_kwargs):
    T, dim = ys.shape
    log_pdf_ys = lambda xs: log_likelihood(xs, ys)
    P0 = F = Q = jnp.eye(dim)
    Q = sigma ** 2 * Q
    P0 = sigma ** 2 * P0

    Fs = jnp.repeat(F[None, ...], T - 1, axis=0)
    Qs = jnp.repeat(Q[None, ...], T - 1, axis=0)
    bs = jnp.zeros((T - 1, dim))
    m0 = jnp.zeros((dim,))

    @jax.jit
    def init(xs):
        return tp.init(xs, log_pdf_ys)

    kernel = tp.get_kernel(m0, P0, Fs, Qs, bs, log_pdf_ys, N)

    return kernel, init


def get_mala_kernel(ys, sigma, N=1, style="marginal", stop_gradient=False, **_kwargs):
    def full_log_pdf(xs):
        if stop_gradient:
            xs = jax.lax.stop_gradient(xs)
        return log_pdf(xs, ys, sigma)

    if style == "marginal":
        kernel = partial(mala.kernel, log_pdf=full_log_pdf, N=N, auxiliary=False)
    elif style == "auxiliary":
        kernel = partial(mala.kernel, log_pdf=full_log_pdf, N=N, auxiliary=True)
    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'marginal', 'auxiliary'")

    @jax.jit
    def init(xs):
        return mala.init(xs, full_log_pdf)

    return kernel, init


def get_csmc_kernel(ys, sigma, N, style="bootstrap", **kwargs):
    T, dx = ys.shape

    if style == "bootstrap":
        def M0_rvs(key, _):
            eps = jax.random.normal(key, (N + 1, dx))
            return sigma * eps

        def Mt_rvs(key, x_t_m_1, _):
            eps = jax.random.normal(key, (N + 1, dx))
            return x_t_m_1 + sigma * eps

        M0_logpdf = lambda x: norm.logpdf(x, scale=sigma).sum()
        M0_logpdf = jnp.vectorize(M0_logpdf, signature="(d)->()")
        Mt_logpdf = lambda x_t_m_1, x_t, _params: norm.logpdf(x_t, x_t_m_1, sigma).sum()
        Mt_logpdf = jnp.vectorize(Mt_logpdf, signature="(d),(d)->()", excluded=(2,))
        Gamma_0 = lambda x: log_potential(x, ys[0]) + M0_logpdf(x)
        Gamma_t = lambda x_t_m_1, x_t, y: log_potential(x_t, y) + Mt_logpdf(x_t_m_1, x_t, None)

    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'bootstrap'")

    M0 = M0_rvs, M0_logpdf
    Mt = Mt_rvs, Mt_logpdf, ys[1:]
    Gamma_t_plus_params = Gamma_t, ys[1:]

    kernel = lambda key, state, *_: csmc.kernel(key, state[0], state[1], M0, Gamma_0, Mt, Gamma_t_plus_params, N=N,
                                                **kwargs)
    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    return kernel, init


def get_tp_csmc_kernel(ys, sigma, N, style="marginal", stop_gradient=False, **kwargs):
    T, dim = ys.shape

    def r0(x):
        if stop_gradient:
            return log_potential(jax.lax.stop_gradient(x), ys[0])
        return log_potential(x, ys[0])

    def rt(_, x, y):
        if stop_gradient:
            return log_potential(jax.lax.stop_gradient(x), y)
        return log_potential(x, y)

    rt_plus_params = rt, ys[1:]

    mut = lambda x, _: x
    P0 = Q = F = jnp.eye(dim)
    Q = sigma ** 2 * Q
    P0 = sigma ** 2 * P0

    Qs = jnp.repeat(Q[None, ...], ys.shape[0] - 1, axis=0)
    b = m0 = jnp.zeros((dim,))

    if style == 'marginal':
        kernel = tpf.get_kernel(m0, P0, r0, mut, Qs, rt_plus_params, N=N, **kwargs)
    elif style == 'filtering':
        kernel = atpf.get_kernel(m0, P0, r0, mut, Qs, rt_plus_params, N=N, **kwargs)
    elif style == 'smoothing':
        kernel = atps.get_kernel(m0, P0, r0, mut, Qs, rt_plus_params, N=N, **kwargs)
    elif style == 'twisted':
        Fs = jnp.repeat(F[None, ...], ys.shape[0] - 1, axis=0)
        bs = jnp.repeat(b[None, ...], ys.shape[0] - 1, axis=0)
        kernel = t_atpf.get_kernel(m0, P0, r0, Fs, bs, Qs, rt_plus_params, N=N, **kwargs)
    else:
        raise NotImplementedError(
            f"Unknown style: {style}, choose from 'marginal', 'filtering', 'smoothing', 'twisted'")

    wrapped_kernel = lambda key, state, delta: kernel(key, state[0], state[1], delta, delta)

    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    return wrapped_kernel, init


def get_mala_csmc_kernel(ys, sigma, N, style="marginal", **kwargs):
    @partial(jnp.vectorize, signature='(d)->()')
    def Gamma_0(x):
        return log_potential(x, ys[0]) + norm.logpdf(x, scale=sigma).sum()

    @partial(jnp.vectorize, signature='(d),(d),(d)->()')
    def Gamma_t(x_t_m_1, x_t, y):
        return log_potential(x_t, y) + norm.logpdf(x_t, x_t_m_1, sigma).sum()

    Gamma_t_plus_params = Gamma_t, ys[1:]

    if style == "filtering":
        kernel = lambda key, state, delta: alf.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params, delta,
                                                      delta, N=N, **kwargs)
    elif style == "smoothing":
        kernel = lambda key, state, delta: als.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params, delta,
                                                      delta, N=N, **kwargs)
    elif style == "marginal":
        kernel = lambda key, state, delta: lf.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params, delta,
                                                     delta, N=N, **kwargs)
    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'marginal', 'filtering', 'smoothing'")
    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    return kernel, init


def get_rw_csmc_kernel(ys, sigma, N, **kwargs):
    kwargs.pop("style")

    @partial(jnp.vectorize, signature='(d)->()')
    def Gamma_0(x):
        return log_potential(x, ys[0]) + norm.logpdf(x, scale=sigma).sum()

    @partial(jnp.vectorize, signature='(d),(d),(d)->()')
    def Gamma_t(x_t_m_1, x_t, y):
        return log_potential(x_t, y) + norm.logpdf(x_t, x_t_m_1, scale=sigma).sum()

    Gamma_t_plus_params = Gamma_t, ys[1:]

    kernel = lambda key, state, delta: rw.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params,
                                                 delta, N=N, **kwargs)

    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    return kernel, init


def get_imh_kernel(ys, sigma, N, **kwargs):
    T, dim = ys.shape

    @partial(jnp.vectorize, signature='(T,d)->()')
    def log_pdf_ys(xs):
        return log_likelihood(xs, ys)

    P0 = F = Q = jnp.eye(dim)
    Q = sigma ** 2 * Q
    P0 = sigma ** 2 * P0

    Fs = jnp.repeat(F[None, ...], T - 1, axis=0)
    Qs = jnp.repeat(Q[None, ...], T - 1, axis=0)
    bs = jnp.zeros((T - 1, dim))
    m0 = jnp.zeros((dim,))

    @jax.jit
    def init(xs):
        return imh.init(xs, log_pdf_ys)

    kernel = imh.get_kernel(m0, P0, Fs, Qs, bs, log_pdf_ys, N)
    kernel_ = lambda key, state, *_: kernel(key, state)
    return kernel_, init
