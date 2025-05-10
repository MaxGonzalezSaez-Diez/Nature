#!/bin/sh
#SBATCH -o "slurm/outfile/%A_%3a"    # $A for the job id, $a for the task id, ie one of [1 - 100]
#SBATCH -e "slurm/errfile/%A_%3a"    # $A for the job id, $a for the task id, ie one of [1 - 100]
#SBATCH --time 36:00:00
#SBATCH -c 8
#SBATCH --mem=80G


source ~/.bashrc
conda activate marl
cd ..
python train.py --config 3sp_compete