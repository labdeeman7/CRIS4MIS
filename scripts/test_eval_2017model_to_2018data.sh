echo "Eval on EndoVis2018"
test_dataset="val"
exp_tag=$1
# Create
# test_path=/scratch/grp/grv_shi/cambridge-1/data/EndoVis2018/${test_dataset}
# pred_path=/scratch/grp/grv_shi/cambridge-1/exp/endovis2017/${exp_tag}/score
# Jade2
test_path=/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2018/${test_dataset}
pred_path=/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/CRIS/exp/endovis2017/${exp_tag}/score

echo "model exp tag: ${exp_tag}"
echo "test path: ${test_path}"
python test.py \
  --config config/endovis2017/${exp_tag}.yaml \
  --only_pred_first_sent \
  --opts TEST.visualize True \
         TEST.test_data_file cris_${test_dataset}.json \
         TEST.test_data_root ./EndoVis2018/${test_dataset}/

echo "eval binary ..."
python evaluate_2018.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type binary

echo "eval parts ..."
python evaluate_2018.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type parts

echo "eval instruments ..."
python evaluate_2018.py \
  --test_path ${test_path} \
  --pred_path ${pred_path} \
  --problem_type instruments