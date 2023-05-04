#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/output_terminal/or_unet_%j.out
#SBATCH --job-name=cris4mis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=262144
#SBATCH --time=2-00:00

source /users/${USER}/.bashrc
source activate MTPSL

python prepare_endovis2018_4_task.py '../data/endovis_2018_all_seq_full_size/val/' 'cris_val_2018.json'