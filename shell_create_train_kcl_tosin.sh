#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/output_terminal/or_unet_%j.out
#SBATCH --job-name=cris4mis
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=262144
#SBATCH --time=2-00:00

source /users/${USER}/.bashrc
source activate MTPSL

echo "first argument is ${1}"

CONFIG=$1
python train.py --config $CONFIG