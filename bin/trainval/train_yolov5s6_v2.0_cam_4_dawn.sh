#!/bin/bash

ROOT_FOLDER=$PWD
export CUDA_VISIBLE_DEVICES=0,1

# Video name
VIDEO_NAME=cam_4_dawn

# Make folder
mkdir bin/trainval/yolov5s6_v2.0_${VIDEO_NAME}

# TRAIN
python bin/trainval/yolov5/train.py  \
    --img 768  \
    --batch 40  \
    --epochs 3000  \
    --project bin/trainval/yolov5s6_v2.0_${VIDEO_NAME}  \
    --data bin/trainval/configs/ai_city_challenge_track_1_${VIDEO_NAME}.yaml  \
    --hyp tss/detector/yolov5/api/data/hyp.finetune.yaml  \
    --weights  bin/trainval/yolov5s6.pt  \
    --infer_weights_path tss/detector/yolov5/weights/yolov5s6/yolov5s6_aicity2021_${VIDEO_NAME}.pt  \
    --infer_weights_full_path tss/detector/yolov5/weights/yolov5s6_trt/yolov5s6_aicity2021_${VIDEO_NAME}_full.pt
    
