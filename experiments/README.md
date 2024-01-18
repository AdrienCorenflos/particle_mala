# Exeriments folder
This folder contains the scripts to run the experiments reported in the paper. 
The `lgssm_folder` contains the scaling experiments for the LGSSM model, both in terms of the number of time steps, the number of dimensions, and the dynamics covariance.
The `stochastic_volatility` contains the comparison between all the methods proposed for different values of the hyperparameters.

In both cases, the experiments were run on a slurm cluster using the `distributed_experiment.sh` file, but the scripts can be easily adapted to run on a local machine by looping over the `distribute.py` file.
To do so, it suffices to run `python distribute.py --i <i>` where `<i>` is the index of the experiment to run among all possible combinations of hyperparameters we considered.
For a list of these combinations, see the `combinations` variable in the `distribute.py` file.
The scripts were set so that they need to be run from inside the folder of the experiment, using a python virtual environment with the library installed.

For a description of the model, we refer to the paper and to the `model.py` file.
The specification of the kernels is in the `kernels.py` file, while the `experiment.py` file contains the actual experiment.

Results are saved in a `results` folder, which is created if it doesn't exist. 
Additionally, a `results_analysis.py` file is provided in the case of the stochastic volatility experiment, which can be used to plot the results and compute the metrics reported in the paper.