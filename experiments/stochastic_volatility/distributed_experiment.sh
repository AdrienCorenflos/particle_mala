#!/bin/sh
#SBATCH -a 0-7
#SBATCH -t 23:59:00
#SBATCH -o logs/128-%a.log
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --constraint=milan

source ~/.bashrc

cd $WRKDIR/mala_csmc/ || exit
source venv/bin/activate

cd experiments/stochastic_volatility || exit
python distribute.py --i $SLURM_ARRAY_TASK_ID
