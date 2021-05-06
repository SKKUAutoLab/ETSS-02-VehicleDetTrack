#!/bin/bash

python aicity2021_multithread_main.py --dataset aicity2021_final_trt

python aicity2021_multithread_main_compress.py --dataset aicity2021_final_trt
