#!/bin/bash

# stop at the first error
set -e

#conda create --name etss-vehicledettrack python=3.10 -y
#conda activate etss-vehicledettrack

# Install the required packages.
pip install poetry==1.2.0  # Python dependency management and packaging made easy.
pip install pylabel==0.1.55  # A simple tool for labeling images for object detection.
pip install pycocotools==2.0.8  # Python API for MS COCO
pip install easydict==1.13  # A lightweight dictionary for Python.
pip install munch==4.0.0  # A dot-accessible dictionary (a la JavaScript objects).
pip install multipledispatch==1.0.0  # A generic function dispatcher in Python.
pip install rich==13.9.4  # Rich is a Python library for rich text and beautiful formatting in the terminal.
pip install pynvml==12.0.0  # Python utilities for the NVIDIA Management Library
pip install torchmetrics==1.6.1  # PyTorch native Metrics
pip install validators==0.34.0  # Python Data Validation for Humans
pip install thop==0.1.1.post2209072238  # A tool to count the FLOPs of PyTorch model.
pip install psutil==7.0.0  # Cross-platform lib for process and system monitoring in Python.
pip install filterpy==1.4.5  # Kalman filtering and optimal estimation library
pip install gdown==5.2.0  # Google Drive Public File/Folder Downloader
pip install tensorboard  # TensorFlow's Visualization Toolkit
pip install yacs==0.1.8  # Yet Another Configuration System
pip install termcolor==2.5.0  # ANSI color formatting for output in terminal
pip install lap==0.5.12  # Linear Assignment Problem solver (LAPJV/LAPMOD).
pip install einops==0.8.1  # A new flavour of deep learning operations
pip install opencv-python==4.12.0.88  # Open Source Computer Vision Library
pip install ordered-enum==0.0.10  # An ordered enum class for Python
pip install loguru==0.7.3 # Python logging made (stupidly) simple.
pip install PyQt6==6.9.1 # Python bindings for the Qt cross-platform application and UI framework.

# Run line by line
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118


# In stall the project in editable mode.
rm -rf poetry.lock
poetry install --extras "dev"
rm -rf poetry.lock

echo "Finish installation successfully!"
