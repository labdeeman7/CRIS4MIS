# endovis 2018
# runai submit cris \
#   -i aicregistry:5000/talabi:latest \
#   --gpu 4 \
#   --memory 40G \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
  # --command -- bash /nfs/home/talabi/CRIS4MIS/scripts/dgx/shell_dgx_train_kcl_tosin.sh '/nfs/home/talabi/CRIS4MIS/config/endovis2018/cris_r50_bigger_size_data_aug.yaml'   

runai submit cris-cross-val-5-times \
  -i aicregistry:5000/talabi:latest \
  --gpu 4 \
  --memory 40G \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --command -- bash /nfs/home/talabi/CRIS4MIS/scripts/dgx/shell_dgx_train_kcl_tosin_5_times.sh '/nfs/home/talabi/CRIS4MIS/config/endovis2018/cris_r50_data_aug_mae_cross_val.yaml'   

# runai submit cris-evaluate \
#   -i aicregistry:5000/talabi:latest \
#   --gpu 1 \
#   --memory 40G \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/CRIS4MIS/scripts/dgx/shell_dgx_produce_test_files_tosin.sh 'CRIS_R50_BIGGER_SIZE_DATA_AUG' 'cris_r50_bigger_size_data_aug' 

# runai submit cris-get-evaluate-tosin \
#   -i aicregistry:5000/talabi:latest \
#   --gpu 1 \
#   --memory 40G \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/CRIS4MIS/scripts/dgx/shell_dgx_evaluate_tosin.sh 'CRIS_R50_BIGGER_SIZE_DATA_AUG' 

