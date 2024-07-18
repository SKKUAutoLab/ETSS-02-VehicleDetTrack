#!/bin/bash

# NOTE: Run each camera
python main.py  \
   --dataset carla_rain  \
   --config Town10HD_location_4.yaml  \
   --write_video True



#python main.py  \
#    --dataset Korea_cctv_rain  \
#    --config 23.yaml  \
#    --write_video True

# NOTE: Run all cameras
#python main_all.py  \
#   --dataset aicity2021_final \
#   --write_video True

# NOTE: Run each camera with multi thread
#python main_multithread.py  \
#    --dataset aicity2021_final \
#    --config cam_1.yaml

# NOTE: Run each camera with multi thread
#python main_all_multithread.py  \
#    --dataset aicity2021_final

# NOTE: Run each camera with multi process
#python main_all_multiprocess.py  \
#    --dataset aicity2021_final

# NOTE: Run each camera without detection
#python main_non_detection.py  \
#    --dataset carla_rain  \
#    --config Town10HD_location_6.yaml  \
#    --write_video True

# NOTE: Run each camera with multi process and  multi thread
#python main_all_multiprocess_multithread.py  \
#    --dataset aicity2021_final \
#    --nodes 1
