#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/output_terminal/cris_%j.out
#SBATCH --job-name=cris4mis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=262144

test_dataset='val'
# Create
test_path=../data/endovis_2018_all_seq_full_size/val
pred_path=exp/endovis2018/CRIS_R50/score

echo "model exp tag: ${exp_tag}"
echo "test path: ${test_path}"


echo "eval binary ..."
python evaluate_tosin.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type binary

echo "eval parts ..."
python evaluate_tosin.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type parts

echo "eval instruments ..."
python evaluate_tosin.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type instruments


echo "eval anatomy ..."
python evaluate_tosin.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type anatomy