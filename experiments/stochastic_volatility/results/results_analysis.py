from functools import reduce
from itertools import product

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow_probability.substrates.jax as tfp
import tqdm.auto as tqdm

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config to load
TARGET = "0.75"
T = 128
D = 30
N = 31

# Config for analysis
ENERGY_SUBSAMPLE = 100  # energy subsampling for plotting
STD_CONFIDENCE = 1.96  # confidence interval for marginals
MAX_LAG = 249  # max lag for autocorrelation
N_EXPERIMENTS = 5  # number of independent experiments we ran
N_SAMPLES = 45_000  # number of samples we took from the posterior
N_CHAINS = 4  # number of chains we ran
TAUS = [10, 50, 100, 200]  # values of tau we ran

MAKE_CSVS = False
PLOT_ENERGY_TRACE = False
PLOT_ENERGY_ACF = False
PLOT_TRAJECTORY = 15
PLOT_ESS = False
PLOT_ACCEPTANCE_RATE = True
PLOT_DELTAS = False


def cross_chains_autocorr(samples, chain_axis):
    # Compute autocorrelation with cross-chain variance reduction.
    # We do this manually because tfp does not expose it as part of the public API.
    # This is a copy-paste of the first part of tfp.mcmc.effective_sample_size with the cross-chain variance reduction.
    auto_cov = tfp.stats.auto_correlation(
        samples, axis=0, max_lags=MAX_LAG, normalize=False)

    between_chain_variance_div_n = tfp.mcmc.diagnostic._reduce_variance(  # noqa
        np.mean(samples, axis=0),
        biased=False,  # This makes the denominator be C - 1.
        axis=chain_axis - 1)

    biased_within_chain_variance = jnp.mean(auto_cov[0], chain_axis - 1)

    approx_variance = biased_within_chain_variance + between_chain_variance_div_n

    mean_auto_cov = jnp.mean(auto_cov, chain_axis)
    auto_corr = 1. - (biased_within_chain_variance - mean_auto_cov) / approx_variance
    return auto_corr


def format_grid(facetgrid):
    for i, axes_row in enumerate(facetgrid.axes):
        for j, axes_col in enumerate(axes_row):
            row, col = axes_col.get_title().split('|')
            col = col.strip().replace("name = ", "")
            row = row.strip().replace("name = ", "")
            col = col.strip().replace("stat = ", "")
            row = row.strip().replace("stat = ", "")

            if i == 0:
                axes_col.set_title(col)
            else:
                axes_col.set_title(f'')

            if j == 0:
                axes_col.set_ylabel(row)


kernel_sytle_lst = [
    ("CSMC", "bootstrap"),
    ("RW_CSMC", "na"),
    ("ADAPTED_CSMC", "filtering"),
    ("ADAPTED_CSMC", "marginal"),
    ("ADAPTED_CSMC", "twisted"),
    ("MALA_CSMC", "filtering"),
    ("MALA_CSMC", "marginal"),
    ("MALA_CSMC", "smoothing"),
    ("TP", "na"),
    ("TP_CSMC", "filtering"),
    ("TP_CSMC", "marginal"),
    ("TP_CSMC", "twisted"),
    ("MALA", "auxiliary"),
    ("MALA", "marginal"),
]
name_in_paper = [
    "CSMC\n",
    "Particle-RWM\n",
    "Particle-aGRAD\n($\\kappa = 0$)",
    "Particle-mGRAD\n($\\kappa = 0$)",
    "Particle-aGRAD\n($\\kappa = 0$, twisted)",
    "Particle-aMALA\n",
    "Particle-MALA\n",
    "Particle-aMALA+\n",
    "aGRAD\n",
    "Particle-aGRAD\n($\\kappa = 1$)",
    "Particle-mGRAD\n($\\kappa = 1$)",
    "Particle-aGRAD\n($\\kappa = 1$, twisted)",
    "aMALA\n",
    "MALA\n",
]

NON_GAUSSIAN_KERNELS = [
    "CSMC\n",
    "Particle-RWM\n",
    "Particle-aMALA\n",
    "Particle-MALA\n",
    "Particle-aMALA+\n",
    "aMALA\n",
    "MALA\n",
]

EXCLUDED_KERNELS = [
    "CSMC\n",
    "aMALA\n",
    "MALA\n",
]

NON_GAUSSIAN_KERNELS_WITHOUT_EXCLUDED = [k for k in NON_GAUSSIAN_KERNELS if k not in EXCLUDED_KERNELS]

GAUSSIAN_KERNELS = [
    "Particle-aGRAD\n($\\kappa = 0$)",
    "Particle-mGRAD\n($\\kappa = 0$)",
    "Particle-aGRAD\n($\\kappa = 1$)",
    "Particle-mGRAD\n($\\kappa = 1$)",
    "aGRAD\n"
]

TWISTED_KERNELS = [
    "aGRAD\n",
    "Particle-aGRAD\n($\\kappa = 0$, twisted)",
    "Particle-aGRAD\n($\\kappa = 1$, twisted)",
]

ALL_GAUSSIAN_KERNELS = GAUSSIAN_KERNELS + [k for k in TWISTED_KERNELS if k not in GAUSSIAN_KERNELS]
name_to_kernel_style = dict(zip(name_in_paper, kernel_sytle_lst))
kernel_sytle_to_name = dict(zip(kernel_sytle_lst, name_in_paper))

experiments_range = range(N_EXPERIMENTS)
chains_range = range(N_CHAINS)
samples_range = range(0, N_SAMPLES, ENERGY_SUBSAMPLE)
lags_range = range(MAX_LAG + 1)
time_range = range(T)
dimensions_range = range(D)

df_columns_index = pd.MultiIndex.from_product([TAUS, name_in_paper], names=["$\\tau$", "name"])

energy_index = pd.MultiIndex.from_product([experiments_range, chains_range, samples_range],
                                          names=["n_experiment", "n_chain", "sample"])
energy_acf_index = pd.MultiIndex.from_product([experiments_range, lags_range],
                                              names=["n_experiment", "lag"])
ess_index = pd.MultiIndex.from_product([("min", "med", "max"), experiments_range, time_range],
                                       names=["stat", "n_experiment", "time"])

true_trajectory_index = pd.MultiIndex.from_product([experiments_range, time_range, dimensions_range],
                                                   names=["n_experiment", "time", "dimension"])

initial_trajectory_index = true_trajectory_index
marginal_mean_index = pd.MultiIndex.from_product([experiments_range, chains_range, time_range, dimensions_range],
                                                 names=["n_experiment", "n_chain", "time", "dimension"])
marginal_conf_index = marginal_mean_index
acceptance_rate_index = pd.MultiIndex.from_product([experiments_range, chains_range, time_range],
                                                   names=["n_experiment", "n_chain", "time"])
delta_index = pd.MultiIndex.from_product([experiments_range, time_range],
                                         names=["n_experiment", "time"])

# dtype = {
#         "$\\tau$": 'int64',
#         "n_experiment": 'int64',
#         "n_chain": 'int64',
#         "sample": 'int64',
#         "lag": 'int64',
#         "time": 'int64',
#         "dimension": 'int64',
#     }

if MAKE_CSVS:

    marginal_mean_df = pd.DataFrame(columns=df_columns_index, index=marginal_mean_index)
    marginal_conf_df_plus = pd.DataFrame(columns=df_columns_index, index=marginal_conf_index)
    marginal_conf_df_minus = pd.DataFrame(columns=df_columns_index, index=marginal_conf_index)
    true_trajectory_df = pd.DataFrame(columns=df_columns_index, index=true_trajectory_index)
    initial_trajectory_df = pd.DataFrame(columns=df_columns_index, index=initial_trajectory_index)

    energy_acfs_df = pd.DataFrame(columns=df_columns_index, index=energy_acf_index)
    energy_data_df = pd.DataFrame(columns=df_columns_index, index=energy_index)

    ess_df = pd.DataFrame(columns=df_columns_index, index=ess_index)
    ess_time_df = pd.DataFrame(columns=df_columns_index, index=ess_index)
    acceptance_rate_df = pd.DataFrame(columns=df_columns_index, index=acceptance_rate_index)
    delta_df = pd.DataFrame(columns=df_columns_index, index=delta_index)

    for column in tqdm.tqdm(product(TAUS, name_in_paper), total=len(TAUS) * len(name_in_paper)):
        tau, name = column
        kernel, style = name_to_kernel_style[name]

        data = np.load(f"kernel={kernel},T={T},D={D},tau={tau},N={N},style={style},target={TARGET}.npz")
        energy_acfs = jax.vmap(cross_chains_autocorr, in_axes=(0, None))(data["energy"], 1)
        energy_acfs = np.asarray(energy_acfs)

        energy_data_df[column].iloc[:] = np.transpose(-data["energy"][:, ::ENERGY_SUBSAMPLE], [0, 2, 1]).flatten()
        energy_acfs_df[column].iloc[:] = energy_acfs.flatten()
        marginal_mean_df[column].iloc[:] = data["means"].flatten()
        marginal_conf_df_plus[column].iloc[:] = data["means"].flatten() + STD_CONFIDENCE * data["std_devs"].flatten()
        marginal_conf_df_minus[column].iloc[:] = data["means"].flatten() - STD_CONFIDENCE * data["std_devs"].flatten()

        true_trajectory_df[column].iloc[:] = data["true_xs"].flatten()
        initial_trajectory_df[column].iloc[:] = data["init_xs"].flatten()

        ess_df[column].loc["min"] = data["ess"].min(-1).flatten()
        ess_df[column].loc["med"] = np.median(data["ess"], -1).flatten()
        ess_df[column].loc["max"] = data["ess"].max(-1).flatten()

        ess_time_df[column].loc["min"] = (data["ess"] / data["sampling_time"][:, None, None]).min(-1).flatten()
        ess_time_df[column].loc["med"] = np.median(data["ess"] / data["sampling_time"][:, None, None], -1).flatten()
        ess_time_df[column].loc["max"] = (data["ess"] / data["sampling_time"][:, None, None]).max(-1).flatten()
        acceptance_rate_df[column].iloc[:] = data["final_pct"].flatten()
        delta_df[column].iloc[:] = data["delta"].flatten()

    ess_df.to_csv("ess.csv")
    ess_time_df.to_csv("ess_time.csv")
    acceptance_rate_df.to_csv("acceptance_rate.csv")
    delta_df.to_csv("delta.csv")
    true_trajectory_df.to_csv("true_trajectory.csv")
    initial_trajectory_df.to_csv("initial_trajectory.csv")
    marginal_mean_df.to_csv("marginal_mean.csv")
    marginal_conf_df_plus.to_csv("marginal_conf_plus.csv")
    marginal_conf_df_minus.to_csv("marginal_conf_minus.csv")
    energy_data_df.to_csv("energy_data.csv")
    energy_acfs_df.to_csv("energy_acfs.csv")

else:


    ess_df = pd.read_csv("ess.csv", header=[0, 1], index_col=[0, 1, 2])
    ess_time_df = pd.read_csv("ess_time.csv", header=[0, 1], index_col=[0, 1, 2])
    acceptance_rate_df = pd.read_csv("acceptance_rate.csv", header=[0, 1], index_col=[0, 1, 2])
    delta_df = pd.read_csv("delta.csv", header=[0, 1], index_col=[0, 1])
    true_trajectory_df = pd.read_csv("true_trajectory.csv", header=[0, 1], index_col=[0, 1, 2])
    initial_trajectory_df = pd.read_csv("initial_trajectory.csv", header=[0, 1], index_col=[0, 1, 2])
    marginal_mean_df = pd.read_csv("marginal_mean.csv", header=[0, 1], index_col=[0, 1, 2, 3])
    marginal_conf_df_plus = pd.read_csv("marginal_conf_plus.csv", header=[0, 1], index_col=[0, 1, 2, 3])
    marginal_conf_df_minus = pd.read_csv("marginal_conf_minus.csv", header=[0, 1], index_col=[0, 1, 2, 3])
    energy_data_df = pd.read_csv("energy_data.csv", header=[0, 1], index_col=[0, 1, 2])
    energy_acfs_df = pd.read_csv("energy_acfs.csv", header=[0, 1], index_col=[0, 1])

    ess_df.set_index(ess_index, inplace=True)
    ess_time_df.set_index(ess_index, inplace=True)
    acceptance_rate_df.set_index(acceptance_rate_index, inplace=True)
    delta_df.set_index(delta_index, inplace=True)
    true_trajectory_df.set_index(true_trajectory_index, inplace=True)
    initial_trajectory_df.set_index(initial_trajectory_index, inplace=True)
    marginal_mean_df.set_index(marginal_mean_index, inplace=True)
    marginal_conf_df_plus.set_index(marginal_conf_index, inplace=True)
    marginal_conf_df_minus.set_index(marginal_conf_index, inplace=True)
    energy_data_df.set_index(energy_index, inplace=True)
    energy_acfs_df.set_index(energy_acf_index, inplace=True)

    ess_df.columns = df_columns_index
    ess_time_df.columns = df_columns_index
    acceptance_rate_df.columns = df_columns_index
    delta_df.columns = df_columns_index
    true_trajectory_df.columns = df_columns_index
    initial_trajectory_df.columns = df_columns_index
    marginal_mean_df.columns = df_columns_index
    marginal_conf_df_plus.columns = df_columns_index
    marginal_conf_df_minus.columns = df_columns_index
    energy_data_df.columns = df_columns_index
    energy_acfs_df.columns = df_columns_index

for tau in TAUS:

    if PLOT_ENERGY_TRACE:
        sns.set(rc={"figure.figsize": (30, 21)})
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")
        sns.set_context("paper", font_scale=2)
        sns.set_style({"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})

        flat_energy_df = energy_data_df.stack([0, 1]).reset_index().rename(columns={0: "neg-energy"})
        flat_energy_df = flat_energy_df[flat_energy_df["$\\tau$"] == tau]
        gaussian_df = flat_energy_df[flat_energy_df["name"].isin(ALL_GAUSSIAN_KERNELS)]
        non_gaussian_df = flat_energy_df[flat_energy_df["name"].isin(NON_GAUSSIAN_KERNELS)]
        # Group both selection as one

        grid = sns.FacetGrid(data=gaussian_df,
                             col="name",
                             row="n_experiment",
                             hue="n_chain",
                             sharex=True,
                             sharey="row",
                             legend_out=True)
        grid.map_dataframe(sns.lineplot, x="sample", y="neg-energy", alpha=1, errorbar=None)  # .set(yscale='log')
        format_grid(grid)
        #  grid.fig.subplots_adjust(top=0.85)
        #  grid.fig.suptitle(f'Gaussian energy trace, $\\tau = {tau}$', size=30)
        plt.savefig(f"figures/gaussian_energy_trace_{tau}.pdf")
        plt.close()

        grid = sns.FacetGrid(data=non_gaussian_df,
                             col="name",
                             row="n_experiment",
                             hue="n_chain",
                             sharex=True,
                             sharey="row",
                             legend_out=True)
        grid.map_dataframe(sns.lineplot, x="sample", y="neg-energy", alpha=1, errorbar=None)  # .set(yscale='log')
        format_grid(grid)
        #  grid.fig.subplots_adjust(top=0.85)
        #  grid.fig.suptitle(f'Non-Gaussian energy trace, $\\tau = {tau}$', size=30)
        plt.savefig(f"figures/non_gaussian_energy_trace_{tau}.pdf")
        plt.close()

    if PLOT_ENERGY_ACF:
        sns.set(rc={"figure.figsize": (30, 21)})
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")
        sns.set_context("paper", font_scale=2)
        sns.set_style({"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})

        flat_energy_df = energy_acfs_df.stack([0, 1]).reset_index().rename(columns={0: "auto-correlation"})
        flat_energy_df = flat_energy_df[flat_energy_df["$\\tau$"] == tau]
        gaussian_df = flat_energy_df[flat_energy_df["name"].isin(ALL_GAUSSIAN_KERNELS)]
        non_gaussian_df = flat_energy_df[flat_energy_df["name"].isin(NON_GAUSSIAN_KERNELS_WITHOUT_EXCLUDED)]
        # Group both selection as one

        grid = sns.FacetGrid(data=gaussian_df,
                             col="name",
                             row="n_experiment",
                             sharex=True,
                             sharey=True,
                             legend_out=True)
        grid.map_dataframe(sns.lineplot, x="lag", y="auto-correlation", alpha=1, errorbar=None)  # .set(yscale='log')
        format_grid(grid)
        #  grid.fig.subplots_adjust(top=0.85)
        #  grid.fig.suptitle(f'Gaussian energy ACF, $\\tau = {tau}$', size=30)
        plt.savefig(f"figures/gaussian_energy_acf_{tau}.pdf")
        plt.close()

        grid = sns.FacetGrid(data=non_gaussian_df,
                             col="name",
                             row="n_experiment",
                             sharex=True,
                             sharey="row",
                             legend_out=True)
        grid.map_dataframe(sns.lineplot, x="lag", y="auto-correlation", alpha=1, errorbar=None)  # .set(yscale='log')
        format_grid(grid)
        #  grid.fig.subplots_adjust(top=0.85)
        #  grid.fig.suptitle(f'Non-Gaussian energy ACF, $\\tau = {tau}$', size=30)
        plt.savefig(f"figures/non_gaussian_energy_acf_{tau}.pdf")
        plt.close()

    if PLOT_TRAJECTORY is not None:
        import warnings

        warnings.filterwarnings("ignore",
                                category=UserWarning)  # it's telling us we are reusing the same color. That's intended
        sns.set(rc={"figure.figsize": (30, 21)})
        sns.set_style("whitegrid")
        palette = sns.color_palette("colorblind", 1)

        sns.set_context("paper", font_scale=2)
        sns.set_style({"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})

        flat_true_trajectory_df = true_trajectory_df.stack([0, 1]).reset_index().rename(columns={0: "true_trajectory"})
        flat_initial_trajectory_df = initial_trajectory_df.stack([0, 1]).reset_index().rename(
            columns={0: "initial_trajectory"})
        flat_mean_df = marginal_mean_df.stack([0, 1]).reset_index().rename(columns={0: "mean"})
        flat_conf_df_plus = marginal_conf_df_plus.stack([0, 1]).reset_index().rename(columns={0: "conf+"})
        flat_conf_df_minus = marginal_conf_df_minus.stack([0, 1]).reset_index().rename(columns={0: "conf-"})

        lst_dfs = [flat_true_trajectory_df, flat_initial_trajectory_df, flat_mean_df, flat_conf_df_plus,
                   flat_conf_df_minus]
        lst_dfs = [k[k["$\\tau$"] == tau] for k in lst_dfs]
        lst_dfs = [k[k["dimension"] == PLOT_TRAJECTORY] for k in lst_dfs]

        gaussian_lst_dfs = [k[k["name"].isin(ALL_GAUSSIAN_KERNELS)] for k in lst_dfs]
        non_gaussian_lst_dfs = [k[k["name"].isin(NON_GAUSSIAN_KERNELS)] for k in lst_dfs]

        gaussian_df = reduce(lambda left, right: pd.merge(left, right, how='inner'), gaussian_lst_dfs)

        non_gaussian_df = reduce(lambda left, right: pd.merge(left, right, how='inner'), non_gaussian_lst_dfs)

        grid = sns.FacetGrid(data=gaussian_df,
                             col="name",
                             row="n_experiment",
                             col_order=ALL_GAUSSIAN_KERNELS,
                             sharex=True,
                             sharey=True,
                             legend_out=True)

        grid.map_dataframe(sns.lineplot, x="time", y="mean", hue="n_chain", alpha=1, errorbar=None, palette=palette)
        grid.map_dataframe(sns.lineplot, x="time", y="conf+", hue="n_chain", alpha=0.75, errorbar=None, linestyle="--",
                           palette=palette)
        grid.map_dataframe(sns.lineplot, x="time", y="conf-", hue="n_chain", alpha=0.75, errorbar=None, linestyle="--",
                           palette=palette)
        grid.map_dataframe(sns.lineplot, x="time", y="true_trajectory", hue="n_chain", alpha=1, errorbar=None, style=10,
                           palette="Greys")
        # grid.map_dataframe(sns.lineplot, x="time", y="initial_trajectory", hue="n_chain", alpha=1, errorbar=None, style=20)

        format_grid(grid)
        #  grid.fig.subplots_adjust(top=0.85)
        #  grid.fig.suptitle(f'Gaussian trajectory plot, $\\tau = {tau}$', size=30)
        plt.savefig(f"figures/gaussian_trajectory_{tau}.pdf")
        plt.close()

        grid = sns.FacetGrid(data=non_gaussian_df,
                             col="name",
                             row="n_experiment",
                             sharex=True,
                             sharey="row",
                             legend_out=True)

        grid.map_dataframe(sns.lineplot, x="time", y="mean", hue="n_chain", alpha=1, errorbar=None, palette=palette)
        grid.map_dataframe(sns.lineplot, x="time", y="conf+", hue="n_chain", alpha=0.75, errorbar=None, linestyle="--",
                           palette=palette)
        grid.map_dataframe(sns.lineplot, x="time", y="conf-", hue="n_chain", alpha=0.75, errorbar=None, linestyle="--",
                           palette=palette)
        grid.map_dataframe(sns.lineplot, x="time", y="true_trajectory", hue="n_chain", alpha=1, errorbar=None, style=1,
                           palette="Greys")
        # grid.map_dataframe(sns.lineplot, x="time", y="initial_trajectory", hue="n_chain", alpha=1, errorbar=None, style=2)
        format_grid(grid)

        #  grid.fig.subplots_adjust(top=0.85)
        #  grid.fig.suptitle(f'Non-Gaussian trajectory plot, $\\tau = {tau}$', size=30)
        plt.savefig(f"figures/non_gaussian_trajectory_{tau}.pdf")
        plt.close()

    if PLOT_ACCEPTANCE_RATE:
        sns.set(rc={"figure.figsize": (30, 21)})
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")
        sns.set_context("paper", font_scale=2)
        sns.set_style({"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})

        flat_acceptance_rate_df = acceptance_rate_df.stack([0, 1]).reset_index().rename(columns={0: "acceptance_rate"})
        flat_acceptance_rate_df = flat_acceptance_rate_df[flat_acceptance_rate_df["$\\tau$"] == tau]

        gaussian_acceptance_rate_df = flat_acceptance_rate_df[
            flat_acceptance_rate_df["name"].isin(ALL_GAUSSIAN_KERNELS)]
        non_gaussian_acceptance_rate_df = flat_acceptance_rate_df[
            flat_acceptance_rate_df["name"].isin(NON_GAUSSIAN_KERNELS)]

        grid = sns.FacetGrid(data=gaussian_acceptance_rate_df,
                             col="name",
                             row="n_experiment",
                             sharex=True,
                             sharey=True,
                             legend_out=True)

        grid.map_dataframe(sns.lineplot, x="time", y="acceptance_rate", alpha=1, errorbar=("se", 2))
        format_grid(grid)
        #  grid.fig.subplots_adjust(top=0.85)
        #  grid.fig.suptitle(f'Gaussian, $\\tau = {tau}$', size=30)

        plt.savefig(f"figures/gaussian_acceptance_rate_{tau}.pdf")
        plt.close()

        grid = sns.FacetGrid(data=non_gaussian_acceptance_rate_df,
                             col="name",
                             row="n_experiment",
                             sharex=True,
                             sharey=True,
                             legend_out=True)

        grid.map_dataframe(sns.lineplot, x="time", y="acceptance_rate", alpha=1, errorbar=("se", 2))
        format_grid(grid)
        #  grid.fig.subplots_adjust(top=0.85)
        #  grid.fig.suptitle(f'non-Gaussian acceptance rate, $\\tau = {tau}$', size=30)

        plt.savefig(f"figures/non_gaussian_acceptance_rate_{tau}.pdf")
        plt.close()

    if PLOT_DELTAS:
        sns.set(rc={"figure.figsize": (30, 21)})
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")
        sns.set_context("paper", font_scale=2)
        sns.set_style({"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})

        flat_delta_df = delta_df.stack([0, 1]).reset_index().rename(columns={0: "delta"})
        flat_delta_df = flat_delta_df[flat_delta_df["$\\tau$"] == tau]

        gaussian_delta_df = flat_delta_df[flat_delta_df["name"].isin(ALL_GAUSSIAN_KERNELS)]
        non_gaussian_delta_df = flat_delta_df[flat_delta_df["name"].isin(NON_GAUSSIAN_KERNELS_WITHOUT_EXCLUDED)]

        grid = sns.FacetGrid(data=gaussian_delta_df,
                             col="name",
                             row="n_experiment",
                             sharex=True,
                             sharey=True,
                             legend_out=True)

        grid.map_dataframe(sns.lineplot, x="time", y="delta", alpha=1, errorbar=None)
        format_grid(grid)
        #  grid.fig.subplots_adjust(top=0.85)
        #  grid.fig.suptitle(f'Gaussian $\\delta_t$, $\\tau = {tau}$', size=30)

        plt.savefig(f"figures/gaussian_delta_{tau}.pdf")
        plt.close()

        grid = sns.FacetGrid(data=non_gaussian_delta_df,
                             col="name",
                             row="n_experiment",
                             sharex=True,
                             sharey=True,
                             legend_out=True)

        grid.map_dataframe(sns.lineplot, x="time", y="delta", alpha=1, errorbar=None)
        format_grid(grid)
        #  grid.fig.subplots_adjust(top=0.85)
        #  grid.fig.suptitle(f'non-Gaussian $\\delta_t$, $\\tau = {tau}$', size=30)

        plt.savefig(f"figures/non_gaussian_delta_{tau}.pdf")
        plt.close()

if PLOT_ESS:
    sns.set(rc={"figure.figsize": (15, 20)})
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    sns.set_context("paper", font_scale=2)
    sns.set_style({"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})
    flat_ess_df = ess_df.stack([0, 1]).reset_index().rename(columns={0: "ess"})
    flat_ess_df["$\\tau$"] = flat_ess_df["$\\tau$"].astype(int)
    flat_ess_df["$\\tau$"] /= 100
    flat_ess_time_df = ess_time_df.stack([0, 1]).reset_index().rename(columns={0: "ess/s"})
    flat_ess_time_df["$\\tau$"] /= 100
    flat_ess_df = flat_ess_df[flat_ess_df["stat"] == "med"]
    flat_ess_time_df = flat_ess_time_df[flat_ess_time_df["stat"] == "med"]

    gaussian_ess_df = flat_ess_df[flat_ess_df["name"].isin(GAUSSIAN_KERNELS)]
    non_gaussian_ess_df = flat_ess_df[flat_ess_df["name"].isin(NON_GAUSSIAN_KERNELS_WITHOUT_EXCLUDED)]
    twisted_ess_df = flat_ess_df[flat_ess_df["name"].isin(TWISTED_KERNELS)]

    gaussian_ess_time_df = flat_ess_time_df[flat_ess_time_df["name"].isin(GAUSSIAN_KERNELS)]
    non_gaussian_ess_time_df = flat_ess_time_df[
        flat_ess_time_df["name"].isin(NON_GAUSSIAN_KERNELS_WITHOUT_EXCLUDED)]
    twisted_ess_time_df = flat_ess_time_df[flat_ess_time_df["name"].isin(TWISTED_KERNELS)]

    grid = sns.FacetGrid(data=gaussian_ess_df,
                         col="name",
                         row="$\\tau$",
                         # hue="n_experiment",
                         sharex=True,
                         sharey="row",
                         col_order=GAUSSIAN_KERNELS,
                         legend_out=True,
                         row_order=[0.1, 0.5, 1.0, 2.0],
                         aspect=1.5)
    # for i in range(grid.axes.shape[0]):
    #     for j in range(1, grid.axes.shape[1] - 1):
    #         grid.axes[i, j].sharey(grid.axes[i, 0])
    #         grid.axes[i, j].tick_params(axis='y', labelleft=False, labelright=False, which="both")
    # for ax in grid.axes[:, -1]:
    #     ax.sharey = False
    #     ax.tick_params(axis='y', labelleft=False, labelright=True, which="both")
    grid.map_dataframe(sns.lineplot, x="time", y="ess", alpha=1, errorbar=("se", 2))  # .set(yscale='log')
    format_grid(grid)

    #  grid.fig.subplots_adjust(top=0.85)
    #  grid.fig.suptitle(f'Gaussian ESS, $\\tau = {tau}$', size=30)
    grid.tight_layout()
    plt.savefig(f"figures/gaussian_ess_med.pdf", )
    plt.close()

    grid = sns.FacetGrid(data=non_gaussian_ess_df,
                         col="name",
                         row="$\\tau$",
                         # hue="n_experiment",
                         sharex=True,
                         sharey="row",
                         legend_out=True,
                         row_order=[0.1, 0.5, 1.0, 2.0],
                         aspect=1.5)

    grid.map_dataframe(sns.lineplot, x="time", y="ess", alpha=1, errorbar=("se", 2))  # .set(yscale='log')
    format_grid(grid)
    #  grid.fig.subplots_adjust(top=0.85)
    #  grid.fig.suptitle(f'non-Gaussian ESS, $\\tau = {tau}$', size=30)
    grid.tight_layout()
    plt.savefig(f"figures/non_gaussian_ess_med.pdf")
    plt.close()

    # twisted kernels

    grid = sns.FacetGrid(data=twisted_ess_df,
                         col="name",
                         row="$\\tau$",
                         # hue="n_experiment",
                         col_order=TWISTED_KERNELS,
                         sharex=True,
                         sharey="row",
                         legend_out=True,
                         row_order=[0.1, 0.5, 1.0, 2.0],
                         aspect=1.5)

    grid.map_dataframe(sns.lineplot, x="time", y="ess", alpha=1, errorbar=("se", 2))  # .set(yscale='log')
    format_grid(grid)
    #  grid.fig.subplots_adjust(top=0.85)
    #  grid.fig.suptitle(f'twisted plot ESS, $\\tau = {tau}$', size=30)
    grid.tight_layout()

    plt.savefig(f"figures/twisted_ess_med.pdf")
    plt.close()

    # Same with ess/s

    grid = sns.FacetGrid(data=gaussian_ess_time_df,
                         col="name",
                         row="$\\tau$",
                         # hue="n_experiment",
                         col_order=GAUSSIAN_KERNELS,
                         sharex=True,
                         sharey="row",
                         legend_out=True,
                         row_order=[0.1, 0.5, 1.0, 2.0],
                         aspect=1.5)
    # for i in range(grid.axes.shape[0]):
    #     for j in range(1, grid.axes.shape[1] - 1):
    #         grid.axes[i, j].sharey(grid.axes[i, 0])
    #         grid.axes[i, j].tick_params(axis='y', labelleft=False, labelright=False, which="both")
    #
    # for ax in grid.axes[:, -1]:
    #     ax.sharey = False
    #     ax.tick_params(axis='y', labelleft=False, labelright=True, which="both")

    grid.map_dataframe(sns.lineplot, x="time", y="ess/s", alpha=1, errorbar=("se", 2))  # .set(yscale='log')
    format_grid(grid)

    #  grid.fig.subplots_adjust(top=0.85)
    #  grid.fig.suptitle(f'Gaussian plot ESS/s, $\\tau = {tau}$', size=30)
    grid.tight_layout()

    plt.savefig(f"figures/gaussian_ess_per_sec_med.pdf")
    plt.close()
    grid.tight_layout()

    grid = sns.FacetGrid(data=non_gaussian_ess_time_df,
                         col="name",
                         row="$\\tau$",
                         # hue="n_experiment",
                         sharex=True,
                         sharey="row",
                         legend_out=True,
                         row_order=[0.1, 0.5, 1.0, 2.0],
                         aspect=2)

    grid.map_dataframe(sns.lineplot, x="time", y="ess/s", alpha=1, errorbar=("se", 2))  # .set(yscale='log')
    format_grid(grid)
    #  grid.fig.subplots_adjust(top=0.85)
    #  grid.fig.suptitle(f'non-Gaussian ESS/s, $\\tau = {tau}$', size=30)
    grid.tight_layout()

    plt.savefig(f"figures/non_gaussian_ess_per_sec_med.pdf")
    plt.close()

    # twisted kernels

    grid = sns.FacetGrid(data=twisted_ess_time_df,
                         col="name",
                         row="$\\tau$",
                         # hue="n_experiment",
                         col_order=TWISTED_KERNELS,
                         sharex=True,
                         sharey="row",
                         legend_out=True,
                         row_order=[0.1, 0.5, 1.0, 2.0],
                         aspect=2)

    grid.map_dataframe(sns.lineplot, x="time", y="ess/s", alpha=1, errorbar=("se", 2))  # .set(yscale='log')
    format_grid(grid)
    #  grid.fig.subplots_adjust(top=0.85)
    #  grid.fig.suptitle(f'twisted ESS/s, $\\tau = {tau}$', size=30)
    grid.tight_layout()

    plt.savefig(f"figures/twisted_ess_per_sec_med.pdf")
    plt.close()
