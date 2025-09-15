import sys
import os
import time
from copy import deepcopy
from functools import partial
from typing import Optional, Union
from timeit import default_timer as timer

import numpy as np
import cv2
from PyQt6 import uic
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *

from tfe.utils.eval_count.eval import evaluate_one_video


class TFE_BUTTON:
	"""
	This class represents the different classes of objects that can be detected by the AI system.
	Each class is represented by an integer value.
	"""
	START    : int = 1
	STOP     : int = 2
	VIDEO    : int = 3
	CONFIG   : int = 4
	RUN_ALL  : int = 5

# NOTE: Fixed dataset information
video_info = {
	"Town10HD_location_2": {
		"frame_num" : 2000, "movement_num": 12, "index_from": 0, "index_to": 2000,
		"video_path": "data/carla_weather/video/Town10HD_location_2.mp4"
	},
	"Town10HD_location_4": {
		"frame_num" : 2000, "movement_num": 6 , "index_from": 0, "index_to": 2000,
		"video_path": "data/carla_weather/video/Town10HD_location_4.mp4"
	},
	"Town10HD_location_6": {
		"frame_num" : 1761, "movement_num": 4 , "index_from": 0, "index_to": 1761,
		"video_path": "data/carla_weather/video/Town10HD_location_6.mp4"
	},
	"Town10HD_location_7": {
		"frame_num" : 2000, "movement_num": 2 , "index_from": 0, "index_to": 2000,
		"video_path": "data/carla_weather/video/Town10HD_location_7.mp4"
	}
}
segment_period = 10
base_factor    = 1.014405


class TFEMainWindow(QMainWindow):

	# MARK: Magic Functions

	def __init__(
			self,
			ui_path : Optional[str] = "tfe.ui",
			*args, **kwargs
	):
		super(TFEMainWindow, self).__init__()
		# Inite
		self.process_start_time    = None
		self.process_end_time      = None
		self.thread_video          = None
		self.thread_video_function = None
		self.run_flag              = False
		self.run_all_video_index   = -1
		self.video_name            = None

		uic.loadUi(ui_path, self)


		# Build layout
		# self.setWindowTitle(window_title)
		self.build_layout()

	def build_layout(self):
		# Build button
		self.button_start.clicked.connect(partial(self.button_released,TFE_BUTTON.START))
		self.button_stop.clicked.connect(partial(self.button_released,TFE_BUTTON.STOP))
		self.button_video.clicked.connect(partial(self.button_released, TFE_BUTTON.VIDEO))
		self.button_run_all.clicked.connect(partial(self.button_released, TFE_BUTTON.RUN_ALL))


	# MARK: Operation function

	@pyqtSlot()
	def open_video_dialog(self):
		video_path = QFileDialog.getOpenFileName(
			self,
			"Open Video File",
			"./data/carla_weather/video",
			"All Files (*);; Python Files (*.py);; PNG Files (*.png)",
		)

		print(f"Process video: {video_path[0]}")
		self.set_video_information(video_path[0])

	def set_video_information(self, video_path):
		basename_noext = os.path.splitext(os.path.basename(video_path))[0]
		self.thread_video_args.dataset = os.path.basename(os.path.dirname(os.path.dirname(video_path)))
		self.thread_video_args.config = os.path.join(os.path.dirname(os.path.dirname(video_path)), "configs",
		                                             f"{basename_noext}.yaml")
		self.thread_video_args.output = os.path.join(os.path.dirname(os.path.dirname(video_path)), "outputs",
		                                             f"{basename_noext}.txt")
		self.thread_video_args.output_run_all = os.path.join(os.path.dirname(os.path.dirname(video_path)), "outputs",
		                                             f"{self.thread_video_args.dataset}.txt")
		self.thread_video_args.groundtruth = os.path.join(os.path.dirname(os.path.dirname(video_path)), "groundtruths",
		                                             f"{basename_noext}.txt")
		self.thread_video_args.visualize = True
		self.thread_video_args.write_video = False
		self.video_name = basename_noext
		self.label_video_name.setText(f"Dataset : {self.thread_video_args.dataset} -- Video: {self.video_name}")

	def update_display_thread(self, create_camera, args):
		# store the function
		self.thread_video_function = create_camera
		# config=camera_hprams, visualize=args.visualize, write_video=args.write_video
		self.thread_video_args     = deepcopy(args)

	def create_display_thread(
			self,
			CameraThread : Union[QThread, None] = None,
			args : Union[dict, None] = None
	):
		if isinstance(CameraThread, QThread):
			# create the video capture thread
			self.thread_video = CameraThread
			# connect its signal to the update_image slot
			# self.thread_video.updateFrame.connect(self.update_main_display)
			# self.thread_video.updateResult.connect(self.update_result)
			self.thread_video.update_information.connect(self.update_information)
		elif callable(CameraThread):  # if this is a function create camera
			self.thread_video = CameraThread(args)
			# connect its signal to the update_image slot
			# self.thread_video.updateFrame.connect(self.update_main_display)
			# self.thread_video.updateResult.connect(self.update_result)
			self.thread_video.update_information.connect(self.update_information)
			self.video_name  = os.path.splitext(os.path.basename(args.config))[0]
			self.label_video_name.setText(f"Dataset : {args.dataset} -- Video: {self.video_name}")

	def button_released(self, button_type: TFE_BUTTON):

		if button_type is TFE_BUTTON.START:
			self.label_status.setText("STATUS: START")
			self.label_status.setStyleSheet('color: green')
			self.button_start.setEnabled(False)
			self.button_stop.setEnabled(False)
			self.button_video.setEnabled(False)
			self.start_process()
			# self.button_stop.setEnabled(True)

		elif button_type is TFE_BUTTON.STOP:
			print("STOP PRESSED")
			self.label_status.setText("STATUS: STOP")
			self.label_status.setStyleSheet('color: red')
			self.button_start.setEnabled(False)
			self.button_stop.setEnabled(False)
			self.button_video.setEnabled(False)
			self.stop_process()
			# self.button_start.setEnabled(True)
			# self.button_video.setEnabled(True)

		elif button_type is TFE_BUTTON.VIDEO:
			self.label_status.setText("STATUS: VIDEO")
			self.label_status.setStyleSheet('color: white')
			self.button_start.setEnabled(False)
			self.button_stop.setEnabled(False)
			self.button_video.setEnabled(False)
			self.stop_process()
			self.open_video_dialog()
			# self.button_video.setEnabled(True)
			# self.button_start.setEnabled(True)

		elif button_type is TFE_BUTTON.CONFIG:
			self.label_status.setText("STATUS: CONFIG")
			self.label_status.setStyleSheet('color: white')
			self.button_start.setEnabled(False)
			self.button_stop.setEnabled(False)
			self.button_video.setEnabled(False)
			self.stop_process()
			# self.button_start.setEnabled(True)

		elif button_type is TFE_BUTTON.RUN_ALL:
			self.label_status.setText("STATUS: RUN_ALL")
			self.label_status.setStyleSheet('color: green')
			self.button_start.setEnabled(False)
			self.button_stop.setEnabled(False)
			self.button_video.setEnabled(False)
			self.button_run_all.setEnabled(False)
			self.run_all_video_index = 0
			self.start_run_all()
			# self.button_start.setEnabled(True)
			# self.button_video.setEnabled(True)
			# self.button_run_all.setEnabled(True)

	def start_run_all(self):
		# NOTE: Fixed dataset list
		# list_video = [
		# 	"data/carla_weather/video/Town10HD_location_2.mp4",
		# 	"data/carla_weather/video/Town10HD_location_4.mp4",
		# 	"data/carla_weather/video/Town10HD_location_6.mp4",
		# 	"data/carla_weather/video/Town10HD_location_7.mp4",
		# ]
		if self.run_all_video_index > -1:
			if self.run_all_video_index == 0:  # mean start run all
				self.wRMSE_array          = []
				self.vehicleNum_array     = []
				self.period_process_array = []

			# self.run_all_video_index show which video is running
			# self.set_video_information(list_video[self.run_all_video_index])
			self.set_video_information(list(video_info.values())[self.run_all_video_index]["video_path"])
			self.button_released(TFE_BUTTON.START)


	def start_process(self):
		self.run_flag = True
		self.process_start_time = timer()
		print("Camera starting...")
		# start the thread
		if self.thread_video is not None:
			self.thread_video.start()
		elif self.thread_video_function is not None:
			self.create_display_thread(self.thread_video_function, self.thread_video_args)
			self.thread_video.start()
		print("Camera running...")

	def stop_process(self):
		self.run_flag = False
		print("Camera stoping")
		# stop the thread
		if self.thread_video is not None:
			self.thread_video.stop()
			self.thread_video.terminate()
		self.thread_video = None
		self.process_end_time = timer()
		# Give time for the thread to finish
		time.sleep(2)
		print("\nCamera stopped...")

		# evaluation
		if self.video_name is not None:
			self.evaluate_one_result()
			self.run_all_video_index += 1
			if self.run_all_video_index == len(list(video_info.values())):
				self.evaluate_all_result()
				self.run_all_video_index = -1
				self.button_run_all.setEnabled(True)

		# check if run all
		if self.run_all_video_index > -1:
			self.start_run_all()

	def update_information(self, information):
		results    = information["result_count"]
		frame      = information["result_frame"]
		is_running = information["is_run"]

		# this is the place to annouce the camera stop
		if not is_running:
			self.label_evaluation.setText(f"Evaluation Result: {self.thread_video_args.output.replace(os.getcwd(), '')}")
			self.button_released(TFE_BUTTON.STOP)  # stop the done process

		# update count result
		self.update_result(results)

		# update display
		self.update_main_display(frame)

	def update_result(self, results):
		if isinstance(results, str):
			self.label_result.setText(results)
		elif isinstance(results, list):
			results_str = ""
			for result in results:
				if "none" not in str.lower(result):
					results_str += f"{result}\n"
			self.label_result.setText(results_str)

	def update_main_display(self, frame):
		if frame is not None:
			cv_img = frame.copy()
			cv_img = cv2.resize(cv_img, (self.display_frame.width(), self.display_frame.height()))
			convert = QImage(cv_img, cv_img.shape[1], cv_img.shape[0], cv_img.strides[0],
			                 QImage.Format.Format_BGR888).copy()  # copy to avoid crash
			self.display_frame.setPixmap(QPixmap.fromImage(convert))

	# MARK: Evaluation

	def evaluate_one_result(self):
		"""Evaluate the camera's performance.
		"""
		# DEBUG:
		# print(f"{self.groundtruth_path=}\n{self.prediction_path=}")

		wRMSE, vehicle_num = evaluate_one_video(
			self.video_name,
			video_info,
			self.thread_video_args.groundtruth,
			self.thread_video_args.output,
			segment_period
		)

		total_process_time = abs(self.process_end_time - self.process_start_time)
		self.wRMSE_array.append(wRMSE)
		self.vehicleNum_array.append(vehicle_num)
		self.period_process_array.append(total_process_time)

		video_total_frame = list(video_info.values())[self.run_all_video_index]["frame_num"]
		score_effetiveness = wRMSE / vehicle_num
		score_efficiency = 1 - (total_process_time * base_factor) / (1.1 * float(video_total_frame))
		score_f1 = 0.3 * score_efficiency + 0.7 * score_effetiveness

		print(f"vehicle_num    : {vehicle_num}\n"
				f"wRMSE        : {wRMSE:.6f}\n"
				f"nwRMSE       : {wRMSE / vehicle_num:.6f}\n"
				f"period       : {total_process_time:.2f}\n"
				f"index_video  : {self.run_all_video_index}\n"
				f"effetiveness : {score_effetiveness:.6f}\n"
				f"efficiency   : {score_efficiency:.6f}\n"
				f"score_f1     : {score_f1:.6f}\n"
               )


		with open(self.thread_video_args.output, "a") as f_append:
			f_append.write(f"effetiveness : {score_effetiveness:.6f}\n"
			               f"efficiency   : {score_efficiency:.6f}\n"
			               f"score_f1     : {score_f1:.6f}\n"
			      )

	def evaluate_all_result(self):
		video_total_frame  = sum([value["frame_num"] for value in video_info.values()])
		total_process_time = sum(self.period_process_array)

		score_effetiveness = sum(self.wRMSE_array) / sum(self.vehicleNum_array)
		score_efficiency = 1 - (total_process_time * base_factor) / (1.1 * float(video_total_frame))
		score_f1 = 0.3 * score_efficiency + 0.7 * score_effetiveness
		print(f"{self.wRMSE_array=} -- {self.vehicleNum_array=} -- {self.period_process_array}")
		print(f"{sum(self.wRMSE_array)=} -- {sum(self.vehicleNum_array)=} -- {sum(self.period_process_array)=}")
		print(f"{score_effetiveness=}")
		print(f"{score_efficiency=}")
		print(f"{score_f1=}")
		print("*********************")
		print("FINISHED")
		print("*********************")
		self.label_evaluation.setText(
			f"Evaluation Result: {os.path.dirname(self.thread_video_args.output.replace(os.getcwd(), ''))}/{self.thread_video_args.dataset}.txt")
		with open(self.thread_video_args.output_run_all, "w") as f_write:
			f_write.write(f"effetiveness : {score_effetiveness:.6f}\n"
			              f"efficiency   : {score_efficiency:.6f}\n"
			              f"score_f1     : {score_f1:.6f}\n"
			               )

def main():
	# Create an instance of QApplication
	# QApplication manages the GUI application's control flow and main settings.
	# sys.argv is a list in Python, which contains the command-line arguments passed to the script.
	app = QApplication(sys.argv)
	# print(QStyleFactory.keys())
	app.setStyle('Windows')

	# Create an instance of our MainWindow class
	window = TFEMainWindow(ui_path="tfe/views/tfe.ui")

	# Show the window on the screen
	window.show()

	# Start the application's event loop
	# The exec() method enters the main loop of the application and waits until exit() is called
	app.exec()


if __name__ == "__main__":
	main()
