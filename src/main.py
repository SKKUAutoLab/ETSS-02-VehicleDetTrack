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

from tfe.camera import Camera
from tfe.utils import data_dir
from tfe.utils import prints
from tfe.utils import process_config


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
	# NOTE: Start timer
	process_start_time = timer()
	camera_start_time  = timer()

	# NOTE: Get camera config
	args          = parser.parse_args()
	config_path   = os.path.join(data_dir, args.dataset, "configs", args.config)
	camera_hprams = process_config(config_path=config_path)

	# NOTE: Define camera
	camera = Camera(config=camera_hprams, visualize=args.visualize, write_video=args.write_video)
	camera_init_time = timer() - camera_start_time

	# NOTE: Process
	camera.run()

	# NOTE: End timer
	total_process_time = timer() - process_start_time
	prints(f"Total processing time: {total_process_time} seconds.")
	prints(f"Camera init time: {camera_init_time} seconds.")
	prints(f"Actual processing time: {total_process_time - camera_init_time} seconds.")


# MARK: - Entry point

if __name__ == "__main__":
	main()
