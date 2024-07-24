#!/bin/bash

# Add python path for MLKit
export PYTHONPATH=$PYTHONPATH:$PWD

START_TIME="$(date -u +%s.%N)"

python projects/tss/tests/mot_evaluation/evaluate_mtmc.py  \
  --gt projects/tss/data/synthehicle_town05/groundtruths/gt.txt  \
  --re projects/tss/data/synthehicle_town05/outputs/mtmc_result.txt  \
  --ou projects/tss/data/synthehicle_town05/outputs/track_mtmc.txt


#python evaluate_mtmc.py  \
#  --re /media/sugarubuntu/DataSKKU2/2_Dataset/AI_City_Challenge/2021/Track_3/AIC21_Track3_MTMC_Tracking/train/S01/c001/gt/gt.txt  \
#  --gt /media/sugarubuntu/DataSKKU2/2_Dataset/AI_City_Challenge/2021/Track_3/AIC21_Track3_MTMC_Tracking/train/S01/c001/gt/gt.txt


END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
