#!/bin/sh
#SBATCH -o logs/debug.log
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --constraint=milan
#SBATCH -p debug

source ~/.bashrc

cd $WRKDIR/mala_csmc/ || exit
source venv/bin/activate

cd experiments/stochastic_volatility || exit
python distribute.py --i 0
