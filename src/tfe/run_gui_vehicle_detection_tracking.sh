#!/bin/bash

#cd /media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/

eval "$(conda shell.bash hook)"

conda activate etss-vehicledettrack

LOG_FILE_PATH="output.log"

# Clear the output log file
echo "" | tee $LOG_FILE_PATH

time python main_gui.py 2>&1 | tee -a $LOG_FILE_PATH
