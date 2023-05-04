#!/usr/bin/python

CONFIG=$1


#make a for loop, change the config name.
for i in 1 2 3 4 5
do
  python /nfs/home/talabi/CRIS4MIS/train.py --config $CONFIG --opts "cross_validation_iteration" ${i}
done
