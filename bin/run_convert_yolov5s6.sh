#!/bin/bash
weight_input_folder=tss/detector/yolov5/weights/yolov5s6_trt
weight_output_folder=tss/detector/yolov5/weights/yolov5s6_trt
build_folder=bin/tensorrtx/yolov5/build
model=s6
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_full.wts $weight_output_folder/yolov5s6_aicity2021_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_1_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_1_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_1_dawn_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_1_dawn_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_1_rain_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_1_rain_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_2_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_2_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_2_rain_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_2_rain_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_3_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_3_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_3_rain_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_3_rain_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_4_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_4_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_4_dawn_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_4_dawn_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_5_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_5_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_5_dawn_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_5_dawn_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_5_rain_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_5_rain_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_6_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_6_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_6_snow_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_6_snow_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_7_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_7_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_7_dawn_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_7_dawn_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_7_rain_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_7_rain_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_8_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_8_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_9_20_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_9_20_full.engine $model
$build_folder/yolov5 -s $weight_input_folder/yolov5s6_aicity2021_cam_15_full.wts $weight_output_folder/yolov5s6_aicity2021_cam_15_full.engine $model
