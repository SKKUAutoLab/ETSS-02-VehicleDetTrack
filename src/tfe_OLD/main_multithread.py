# ==================================================================== #
# Copyright (C) 2022 - Automation Lab - Sungkyunkwan University
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# ==================================================================== #
import argparse
import os
from timeit import default_timer as timer

from tfe.camera import CameraMultithread
from tfe.io import compress_all_result
from tfe.utils import data_dir
from tfe.utils import prints
from tfe.utils import process_config

import torch

# MARK: - Args

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument(
	"--dataset",
	default="aicity2021_final",
	help="The dataset to run on."
)
parser.add_argument(
	"--queue_size",
	default=10,
	type=int,
	help="The max queue size"
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

parser.add_argument(
	"--config",
	default="Town10HD_location_1.yaml",
	help="The config file for each camera. The final path to the config file is: TSS/data/[dataset]/configs/[config]/"
)


# MARK: - Main Function

def main():
	args = parser.parse_args()

	# NOTE: Start timer
	process_start_time = timer()
	total_camera_init_time = 0

	camera_start_time = timer()

	# NOTE: Get camera config
	config_path   = os.path.join(data_dir, args.dataset, "configs", args.config)
	camera_hprams = process_config(config_path=config_path)

	# NOTE: Define camera
	camera = CameraMultithread(
		config      = camera_hprams,
		queue_size  = args.queue_size,
		visualize   = args.visualize,
		write_video = args.write_video
	)
	total_camera_init_time += timer() - camera_start_time
	# NOTE: Process
	camera.run()

	# NOTE: End timer
	total_process_time = timer() - process_start_time
	prints(f"Total processing time: {total_process_time} seconds.")
	prints(f"Total camera init time: {total_camera_init_time} seconds.")
	prints(f"Actual processing time: {total_process_time - total_camera_init_time} seconds.")


# # NOTE: Compress result from tss.io import compress_all_result
# print("Compressing result")
# output_dir = os.path.join(data_dir, args.dataset, "outputs")
# compress_all_result(output_dir=output_dir)


# MARK: - Entry point

if __name__ == "__main__":
	main()
