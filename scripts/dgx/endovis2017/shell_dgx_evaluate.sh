#!/usr/bin/python
# Create

echo "first argument should be the exp_name ${1}"
exp_name=$1

test_path=/nfs/home/talabi/data/endovis_2017_processed/cropped_test/
pred_path=/nfs/home/talabi/CRIS4MIS/exp/endovis2017/${exp_name}/score

echo "test path: ${test_path}"


echo "eval binary ..."
python /nfs/home/talabi/CRIS4MIS/evaluate.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type binary

echo "eval parts ..."
python /nfs/home/talabi/CRIS4MIS/evaluate.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type parts

echo "eval instruments ..."
python /nfs/home/talabi/CRIS4MIS/evaluate.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type instruments