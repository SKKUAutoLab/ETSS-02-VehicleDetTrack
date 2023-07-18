#!/bin/bash

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")                    # src/motordriver
export DIR_TSS=$DIR_CURRENT                         # src/motordriver
export DIR_SOURCE=$DIR_TSS"/motordriver"            # src/motordriver/motordriver

# Add data dir
export DIR_DATA="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5_test_docker/images"

# Add python path
export PYTHONPATH=$PYTHONPATH:$PWD                              # .
export PYTHONPATH=$PYTHONPATH:$DIR_SOURCE                       # src/motordriver/motordriver
export PYTHONPATH=$PYTHONPATH:$DIR_SOURCE/detectors/ultralytics # src/motordriver/motordriver

export CUDA_LAUNCH_BLOCKING=1

START_TIME="$(date -u +%s.%N)"
###########################################################################################################

# NOTE: COPY FILE
cp -f $DIR_TSS"/configs/class_labels_1cls.json" $DIR_DATA"/class_labels_1cls.json"
cp -f $DIR_TSS"/configs/class_labels_7cls.json" $DIR_DATA"/class_labels_7cls.json"

# NOTE: DETECTION
echo "*********"
echo "RUNNING"
echo "*********"
python $DIR_TSS/main.py  \
  --config aic23.yaml  \
  --run_image

###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
