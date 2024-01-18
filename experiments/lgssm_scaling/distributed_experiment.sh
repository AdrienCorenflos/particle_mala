#!/bin/sh
#SBATCH -a 0-1819
#SBATCH -t 01:59:00
#SBATCH -o logs/%a.log
#SBATCH --mem=8G

source ~/.bashrc

cd $WRKDIR/mala_csmc/ || exit
source venv/bin/activate

cd experiments/lgssm_scaling || exit
python distribute.py --i $SLURM_ARRAY_TASK_ID
