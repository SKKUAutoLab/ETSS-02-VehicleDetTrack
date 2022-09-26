# ==================================================================== #
# File name: main.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/27/2021
#
# The main run script.
# ==================================================================== #
import argparse
import os
from timeit import default_timer as timer

from tss.camera import Camera
from tss.utils import data_dir
from tss.utils import prints
from tss.utils import process_config


# MARK: - Args

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument(
	"--dataset",
	default="carla",
	help="The dataset to run on."
)
parser.add_argument(
	"--config",
	default="Town10HD_location_1.yaml",
	help="The config file for each camera. The final path to the config file is: TSS/data/[dataset]/configs/[config]/"
)
parser.add_argument(
	"--visualize",
	default=False,
	help="Should visualize the processed images"
)
parser.add_argument(
	"--write_video",
	default=False,
	help="Should write processed images to video"
)


# MARK: - Main Function

def main():
	# TODO: Start timer
	process_start_time = timer()
	camera_start_time  = timer()

	# TODO: Get camera config
	args          = parser.parse_args()
	config_path   = os.path.join(data_dir, args.dataset, "configs", args.config)
	camera_hprams = process_config(config_path=config_path)

	# TODO: Define camera
	camera = Camera(config=camera_hprams, visualize=args.visualize, write_video=args.write_video)
	camera_init_time = timer() - camera_start_time

	# TODO: Process
	camera.run()

	# TODO: End timer
	total_process_time = timer() - process_start_time
	prints(f"Total processing time: {total_process_time} seconds.")
	prints(f"Camera init time: {camera_init_time} seconds.")
	prints(f"Actual processing time: {total_process_time - camera_init_time} seconds.")


# MARK: - Entry point

if __name__ == "__main__":
	main()
