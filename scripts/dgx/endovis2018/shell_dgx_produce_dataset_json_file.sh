#!/usr/bin/python
echo "first argument is ${1}"

CONFIG=$1
python  /nfs/home/talabi/CRIS4MIS/tools/prepare_endovis2017.py /nfs/home/talabi/data/endovis_2017_processed/cropped_train cris_test.json 
# python  /nfs/home/talabi/CRIS4MIS/tools/prepare_endovis2017.py --config $CONFIG