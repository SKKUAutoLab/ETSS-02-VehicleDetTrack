#!/bin/bash

# Run each camera
python main.py  \
   --dataset aicity2021_final  \
   --config cam_1.yaml  \
   --write_video True

# Run all cameras
#python main_all.py  \
#   --dataset aicity2021_final \
#   --write_video True

# Run each camera with multi thread
#python main_multithread.py  \
#    --dataset aicity2021_final \
#    --config cam_1.yaml

# Run each camera with multi thread
#python main_all_multithread.py  \
#    --dataset aicity2021_final

# Run each camera with multi process
#python main_all_multiprocess.py  \
#    --dataset aicity2021_final

# Run each camera without detection
#python main_non_detection.py  \
#    --dataset carla  \
#    --config Town10HD_location_1.yaml \
#    --write_video True

# Run each camera with multi process and  multi thread
#python main_all_multiprocess_multithread.py  \
#    --dataset aicity2021_final \
#    --nodes 1
