#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import os
import sys
from timeit import default_timer as timer
from time import perf_counter

import yaml

from core.utils.rich import console
from motordriver.cameras import TrafficSafetyCamera

from configuration import (
	data_dir,
	config_dir,
	load_config
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# MARK: - Args

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument(
	"--config", default="aic23.yaml",
	help="Config file for each camera. Final path to the config file."
)
parser.add_argument(
	"--dataset", default="aic23_trafficsafety",
	help="Dataset to run on."
)
parser.add_argument(
	"--run_image", action='store_true', help="Should run detection."
)
parser.add_argument(
	"--write_final", action='store_true', help="Should run detection."
)
parser.add_argument(
	"--verbose", action='store_true', help="Should visualize the images."
)

Camera = TrafficSafetyCamera


# MARK: - Main Function

def main():
	# NOTE: Start timer
	process_start_time = perf_counter()
	camera_start_time  = perf_counter()

	# NOTE: Parse camera config
	args        = parser.parse_args()
	config_path = os.path.join(config_dir, args.config)
	camera_cfg  = load_config(config_path)

	# NOTE: Update value from args
	camera_cfg["dataset"]      = args.dataset
	camera_cfg["verbose"]      = args.verbose
	camera_cfg["process"]      = {
		"run_image"            : args.run_image,       # All run with image, not video
		"function_write_final" : args.write_final,     # Writing final results.
	}

	# NOTE: Define camera
	camera           = Camera(**camera_cfg)
	camera_init_time = perf_counter() - camera_start_time

	# NOTE: Process
	camera.run()

	# NOTE: End timer
	total_process_time = perf_counter() - process_start_time
	console.log(f"Total processing time: {total_process_time} seconds.")
	console.log(f"Camera init time: {camera_init_time} seconds.")
	console.log(f"Actual processing time: "
				f"{total_process_time - camera_init_time} seconds.")


# MARK: - Entry point

if __name__ == "__main__":
	main()
