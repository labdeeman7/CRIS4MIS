# endovis 2017

# runai submit cris-cross-val \
#   -i aicregistry:5000/talabi:latest \
#   --gpu 4 \
#   --memory 40G \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/CRIS4MIS/scripts/dgx/endovis2017/shell_dgx_train_kcl_tosin_5.sh '/nfs/home/talabi/CRIS4MIS/config/endovis2017/cris_r50_size_416_mae_cross_val.yaml'  
  
# run three experiments, run endovis2017 normal, run endovis2017 mae, mae_shared_encoder, run endovis2017 mae 5 times. 
#get cris_test-files 
# runai submit cris-2017-size-416 \
#   -i aicregistry:5000/talabi:latest \
#   --gpu 4 \
#   --memory 40G \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/CRIS4MIS/scripts/dgx/endovis2017/shell_dgx_train_kcl_tosin_2017.sh '/nfs/home/talabi/CRIS4MIS/config/endovis2017/cris_r50_size_416.yaml' 


# runai submit cris-2017-size-416-mae \
#   -i aicregistry:5000/talabi:latest \
#   --gpu 4 \
#   --memory 40G \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/CRIS4MIS/scripts/dgx/endovis2017/shell_dgx_train_kcl_tosin_2017.sh '/nfs/home/talabi/CRIS4MIS/config/endovis2017/cris_r50_size_416_mae.yaml'   

# runai submit cris \
#   -i aicregistry:5000/talabi:latest \
#   --gpu 4 \
#   --memory 40G \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
  # --command -- bash /nfs/home/talabi/CRIS4MIS/scripts/dgx/shell_dgx_train_kcl_tosin.sh '/nfs/home/talabi/CRIS4MIS/config/endovis2017/cris_r50_bigger_size_data_aug.yaml'    

# runai submit cris-produce-test-files-2017 \
#   -i aicregistry:5000/talabi:latest \
#   --gpu 1 \
#   --memory 40G \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/CRIS4MIS/scripts/dgx/endovis2017/shell_dgx_produce_test_files_tosin.sh

# runai submit cris-evaluate-2017 \
#   -i aicregistry:5000/talabi:latest \
#   --gpu 2 \
#   --memory 40G \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/CRIS4MIS/scripts/dgx/endovis2017/shell_dgx_evaluate.sh 'CRIS_416' 

runai submit cris-mae-evaluate-2017-cross-val \
  -i aicregistry:5000/talabi:latest \
  --gpu 2 \
  --memory 40G \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --command -- bash /nfs/home/talabi/CRIS4MIS/scripts/dgx/endovis2017/shell_dgx_evaluate_cross_val.sh 'CRIS_416_MAE'   



