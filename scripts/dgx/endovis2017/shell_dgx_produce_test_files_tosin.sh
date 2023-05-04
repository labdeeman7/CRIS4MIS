#!/usr/bin/python
exp_yaml_name=cris_416_mae_cross_val

for i in 1 2 3 4 5
do
  python /nfs/home/talabi/CRIS4MIS/test.py \
    --config  /nfs/home/talabi/CRIS4MIS/config/endovis2017/${exp_yaml_name}.yaml \
    --only_pred_first_sent \
    --opts TEST.visualize True \
          TEST.test_data_file cris_test.json \
          TEST.test_data_root  /nfs/home/talabi/data/endovis_2017_processed/cropped_test/ \
          "cross_validation_iteration" ${i}

done