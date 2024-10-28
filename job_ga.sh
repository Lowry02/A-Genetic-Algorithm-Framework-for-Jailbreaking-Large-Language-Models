#!/bin/bash
#SBATCH -A uTS24_Bonin
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --time 12:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=adversarial_ga
#SBATCH --output=job-%x.out

source ~/.bashrc
conda activate adga

python genetic_algorithm.py --batch_size=30 --model_name="meta-llama/Llama-3.2-3B-Instruct" --log_name="llama3.2-3B"
