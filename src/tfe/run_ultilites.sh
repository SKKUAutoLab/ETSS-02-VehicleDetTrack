#!/bin/bash

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate etss-vehicledettrack

#time python ultilities/convert_separate_tracking_result_into_rmois.py  \
#  --input_image_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_result/12_01_SUWON/images"  \
#  --json_rmois "/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/data/Korea_cctv/rmois/12_01_SUWON.json"  \
#  --input_tracking_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_result/12_01_SUWON/labels"  \
#  --output_tracking_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_labels/12_01_SUWON"

#time python ultilities/convert_separate_tracking_result_into_rmois.py  \
#  --input_image_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_result/12_02_SUWON/images"  \
#  --json_rmois "/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/data/Korea_cctv/rmois/12_02_SUWON.json"  \
#  --input_tracking_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_result/12_02_SUWON/labels"  \
#  --output_tracking_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_labels/12_02_SUWON"

#time python ultilities/convert_separate_tracking_result_into_rmois.py  \
#  --input_image_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_result/23_SUWON/images"  \
#  --json_rmois "/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/data/Korea_cctv/rmois/23_SUWON.json"  \
#  --input_tracking_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_result/23_SUWON/labels"  \
#  --output_tracking_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_labels/23_SUWON"
  
#time python ultilities/convert_separate_tracking_result_into_rmois.py  \
#  --input_image_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_result/34_SUWON/images"  \
#  --json_rmois "/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/data/Korea_cctv/rmois/34_SUWON.json"  \
#  --input_tracking_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_result/34_SUWON/labels"  \
#  --output_tracking_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_labels/34_SUWON"

time python ultilities/convert_separate_tracking_result_into_rmois.py  \
  --input_image_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_result/30_SEOUL/images"  \
  --json_rmois "/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/data/Korea_cctv/rmois/30_SEOUL.json"  \
  --input_tracking_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_result/30_SEOUL/labels"  \
  --output_tracking_folder "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_labels/30_SEOUL"

#  --input_video "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/videos/12_02_SUWON.mp4"  \
