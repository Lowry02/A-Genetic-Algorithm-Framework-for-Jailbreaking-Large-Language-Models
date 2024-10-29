#!/bin/bash
#SBATCH -A uTS24_Bonin
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --time 02:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=resp_ga
#SBATCH --output=job-%x.out

source ~/.bashrc
conda activate adga

python get_responses.py


