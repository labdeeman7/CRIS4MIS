#!/usr/bin/python

test_dataset='val'
# Create
echo "first argument should be the exp_name ${1}"

exp_name=$1

test_path=/nfs/home/talabi/data/endovis_2018_all_seq_full_size/val
pred_path=/nfs/home/talabi/CRIS4MIS/exp/endovis2018/${exp_name}/score

echo "test path: ${test_path}"


echo "eval binary ..."
python /nfs/home/talabi/CRIS4MIS/evaluate_tosin.py \ 
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type binary

echo "eval parts ..."
python /nfs/home/talabi/CRIS4MIS/evaluate_tosin.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type parts

echo "eval instruments ..."
python /nfs/home/talabi/CRIS4MIS/evaluate_tosin.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type instruments


echo "eval anatomy ..."
python /nfs/home/talabi/CRIS4MIS/evaluate_tosin.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type anatomy


echo "eval endovis_2018_style ..."
python /nfs/home/talabi/CRIS4MIS/evaluate_tosin.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type endovis_2018_style 