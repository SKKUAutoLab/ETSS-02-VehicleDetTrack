#!/bin/bash

#cd /media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate etss-vehicledettrack

# Define the log file path
LOG_FILE_PATH="/home/sugarubuntu/Downloads/output.log"

# Clear the output log file
echo "" | tee $LOG_FILE_PATH

# Run the Python script and append both stdout and stderr to the log file
time python main.py  \
  --dataset Korea_cctv_rain  \
  --config 23.yaml  \
  --write_video True  \
  2>&1 | tee -a $LOG_FILE_PATH

#time python main_gui.py  2>&1 | tee -a $LOG_FILE_PATH