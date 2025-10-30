import argparse
import os
from timeit import default_timer as timer

from tfe.cameras.camera_qt6 import CameraQT6
from tfe.configuration import data_dir
from tfe.utils.config import process_config
from tfe.views.traffic_flow_estimation_view import *

from loguru import logger

# MARK: - Args

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument(
	"--dataset",
	default="carla_weather",
	help="The dataset to run on."
)
parser.add_argument(
	"--config",
	default="Town10HD_location_2.yaml",
	help="The config file for each camera. The final path to the config file is: TSS/data/[dataset]/configs/[config]/"
)
parser.add_argument(
	"--visualize",
	default=True,
	help="Should visualize the processed images"
)
parser.add_argument(
	"--write_video",
	default=False,
	help="Should write processed images to video"
)


def create_camera(args):
	camera_hprams = process_config(config_path=args.config)
	return CameraQT6(config=camera_hprams, visualize=args.visualize, write_video=args.write_video)


def main():
	# NOTE: Start timer
	process_start_time = timer()

	# NOTE: Get camera config
	args = parser.parse_args()
	args.config = os.path.join(data_dir, args.dataset, "configs", args.config)

	# NOTE: Define camera
	# camera = create_camera(args)

	# NOTE: create GUI
	# Create an instance of QApplication
	app = QApplication(sys.argv)
	app.setStyle('Fusion')
	# print(QStyleFactory.keys())
	# app.setStyle('Windows')

	# Create an instance of our MainWindow class
	window = TFEMainWindow(ui_path="tfe/views/tfe.ui")

	# Camera for WINDOW GUI
	# window.create_display_thread(camera)
	# window.create_display_thread(
	# 	partial(
	# 		create_camera,
	# 		args
	# 	)
	# )
	window.update_display_thread(create_camera, args)

	# Show the window on the screen
	window.show()

	# NOTE: Process
	# Start the application's event loop
	app.exec()

	# NOTE: End timer
	total_process_time = timer() - process_start_time
	logger.success(f"Total processing time: {total_process_time} seconds.")


if __name__ == "__main__":
	main()
