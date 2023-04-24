#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/output_terminal/cris_%j.out
#SBATCH --job-name=cris4mis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=262144

source /users/${USER}/.bashrc
source activate MTPSL

echo "exp_name is ${1}"
echo "exp_yaml_name is ${2}"

exp_name=$1
exp_yaml_name=$2

test_dataset='val'
# exp_tag=cris_r50_data_aug
# Create
test_path=../data/endovis_2018_all_seq_full_size/val
pred_path=exp/endovis2018/${exp_name}/score

echo "model exp tag: ${exp_name}"
echo "test path: ${test_path}"

python test.py \
  --config config/endovis2018/${exp_yaml_name}.yaml \
  --only_pred_first_sent \
  --opts TEST.visualize True \
         TEST.test_data_file cris_${test_dataset}_2018.json \
         TEST.test_data_root ../data/endovis_2018_all_seq_full_size/val/