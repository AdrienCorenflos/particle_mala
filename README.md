# Gradient-based conditional SMC for inference of high-dimensional Markovian systems
-----------------
This repository contains the code for the paper [Particle-MALA and Particle-mGRAD: Gradient-based MCMC methods for high-dimensional state-space models](https://arxiv.org/abs/2401.14868) by [Adrien Corenflos](https://adriencorenflos.github.io/) and [Axel Finke](https://www.lboro.ac.uk/departments/maths/staff/axel-finke/), see also the attached [preprint](https://github.com/AdrienCorenflos/particle_mala/blob/main/preprint.pdf).
It contains both a general implementation of the algorithms considered and the code to reproduce the experiments.

## Abstract

State-of-the-art methods for Bayesian inference in state-space models are (a) [conditional sequential Monte Carlo (CSMC)](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2009.00736.x) algorithms; (b) sophisticated ‘classical’ MCMC algorithms like MALA, or mGRAD from [Titsias and Papaspiliopoulos (2018)](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12269). 
The former propose $N$ particles at each time step to exploit the model's ‘decorrelation-over-time’ property and thus scale favourably with the time horizon, $T$, but break down if the dimension of the latent states, $D$, is large. 
The latter leverage gradient- or prior-informed local proposals to scale favourably with $D$ but exhibit sub-optimal scalability with $T$ due to a lack of model-structure exploitation. 

We introduce methods which combine the strengths of both approaches. 

The first, Particle-MALA, spreads $N$ particles locally around the current state using gradient information, thus extending MALA to $T > 1$ time steps and $N > 1$ proposals. The second, Particle-mGRAD, additionally incorporates (conditionally) Gaussian prior dynamics into the proposal, thus extending the mGRAD algorithm to $T > 1$ time steps and $N > 1$ proposals. We prove that Particle-mGRAD interpolates between CSMC and Particle-MALA, resolving the ‘tuning problem’ of choosing between CSMC (superior for highly informative prior dynamics) and Particle-MALA (superior for weakly informative prior dynamics). 

We similarly extend other ‘classical’ MCMC approaches like auxiliary MALA, aGRAD, and [preconditioned Crank–Nicolson–Langevin (PCNL)](https://projecteuclid.org/journals/statistical-science/volume-28/issue-3/MCMC-Methods-for-Functions--Modifying-Old-Algorithms-to-Make/10.1214/13-STS421.full) to $T > 1$ time steps and $N > 1$ proposals. 
In experiments, for both highly and weakly informative prior dynamics, our methods substantially improve upon both CSMC and sophisticated ‘classical’ MCMC approaches.

## Expected behaviour
|  ![lgssm_dimension](https://github.com/AdrienCorenflos/particle_mala/assets/19948263/5980e1ae-16b8-4857-8fdf-9643255216fc) | ![lgssm_time](https://github.com/AdrienCorenflos/particle_mala/assets/19948263/b4336faf-4c82-44b6-9ffb-f8cfc44a8f92) | ![lgssm_variance](https://github.com/AdrienCorenflos/particle_mala/assets/19948263/111ff857-0c6d-4d1f-aba0-d405e1ac620c) |
| :--: | :--: | :--: | 
| *Scaling of the methods with the dimension* | *Scaling of the methods with the number of time steps* | *Scaling for informative vs. uninformative dynamics* |

## Installation
Create a clean environment using your favourite environment manager.
Run `pip install .` (or if you use `poetry` follow their instructions) in the root folder of the repository.

## Usage

The following methods are given in order of decreasing generality: `1.` applies to all the models, `2.` does not apply to `1.`, `3.` does not apply to `1.` and `2.`, etc.
1. The system is simply defined as a product of bi-variate potential functions. This is the most general case, the user can use one of the following methods:
    - `rw_csmc.py` for the Particle-RWM algorithm.
    - `al_csmc_f.py` (auxiliary Langevin CSMC with filtering gradients) named Particle-aMALA in the paper
    - `al_csmc_s.py` (auxiliary Langevin CSMC with smoothing gradients) named Particle-aMALA+ in the paper
    - `l_csmc_f.py` (marginal MALA CSMC with filtering gradients) named Particle-MALA in the paper
2. The system is given as a generic Feynman–Kac model. In this case, should simply use the `csmc.py` file.
3. The system is given as a Feynman–Kac model with conditionally Gaussian dynamics. In this case, the user can use
    - `atp_csmc_f.py` (auxiliary Titsias–Papaspiliopoulos CSMC with filtering gradients) named Particle-aGRAD in the paper
    - `atp_csmc_s.py` (auxiliary Titsias–Papaspiliopoulos CSMC with smoothing gradients) named Particle-aGRAD+ in the paper. Note that this recovers Particle-aGRAD if the potentials are univariate.
    - `tp_csmc_f.py` (marginal Titsias–Papaspiliopoulos CSMC with filtering gradients) named Particle-mGRAD in the paper
4. The system is given as a Feynman–Kac model with fully Gaussian dynamics. In this case, the user can use
    - `t_atp_csmc_f.py` (twisted auxiliary Titsias–Papaspiliopoulos CSMC with filtering gradients) named twisted Particle-aGRAD in the paper

Additionally to these, a MALA algorithm (potentially with multiple proposals) is given in `mala.py` and an aGRAD algorithm (potentially with multiple proposals) is given in `grad.py`.
For a description of the algorithms, we refer to our article, and for a description of the arguments of each method, to the corresponding file.

## Reproducing the experiments
Details are given in the `README` file in the corresponding folder.
