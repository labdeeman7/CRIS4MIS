#!/usr/bin/python

source /users/${USER}/.bashrc
source activate MTPSL

echo "exp_name is ${1}"
echo "exp_yaml_name is ${2}"

exp_name=$1
exp_yaml_name=$2

test_dataset='val'

echo "model exp tag: ${exp_name}"
echo "test path: ${test_path}"

python /nfs/home/talabi/CRIS4MIS/test.py \
  --config  /nfs/home/talabi/CRIS4MIS/config/endovis2018/${exp_yaml_name}.yaml \
  --only_pred_first_sent \
  --opts TEST.visualize True \
         TEST.test_data_file cris_${test_dataset}_2018.json \
         TEST.test_data_root  /nfs/home/talabi/data/endovis_2018_all_seq_full_size/val/