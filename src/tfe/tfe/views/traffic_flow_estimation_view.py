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

from tfe.utils.eval_nwrmse import evaluate_one_video

from loguru import logger

class TFE_BUTTON:
	"""
	This class represents the different classes of objects that can be detected by the AI system.
	Each class is represented by an integer value.
	"""
	START  : int = 1
	STOP   : int = 2
	VIDEO  : int = 3
	CONFIG : int = 4
	RUN_ALL: int = 5

# NOTE: Fixed dataset information
video_info_carla = {
	"Town10HD_location_2": {
		"frame_num" : 2000, "movement_num": 12, "index_from": 0, "index_to": 2000, "number_of_classes": 2,
		"video_path": "data/carla_weather/video/Town10HD_location_2.mp4"
	},
	"Town10HD_location_4": {
		"frame_num" : 2000, "movement_num": 6 , "index_from": 0, "index_to": 2000, "number_of_classes": 2,
		"video_path": "data/carla_weather/video/Town10HD_location_4.mp4"
	},
	"Town10HD_location_6": {
		"frame_num" : 1761, "movement_num": 4 , "index_from": 0, "index_to": 1761, "number_of_classes": 2,
		"video_path": "data/carla_weather/video/Town10HD_location_6.mp4"
	},
	"Town10HD_location_7": {
		"frame_num" : 2000, "movement_num": 2 , "index_from": 0, "index_to": 2000, "number_of_classes": 2,
		"video_path": "data/carla_weather/video/Town10HD_location_7.mp4"
	}
}

video_info_Korea_cctv_folders = {
	"23_SUWON": {
		"frame_num" : 6001, "movement_num": 10, "index_from": 0, "index_to": 6000, "number_of_classes": 8,
		"video_path": "data/Korea_cctv/video/23_SUWON"
	},
	"34_SUWON": {
		"frame_num" : 6001, "movement_num": 11 , "index_from": 0, "index_to": 6000, "number_of_classes": 8,
		"video_path": "data/Korea_cctv/video/34_SUWON"
	}
}

video_info_Korea_cctv_files = {
	"23_SUWON": {
		"frame_num" : 6001, "movement_num": 10, "index_from": 0, "index_to": 6000, "number_of_classes": 8,
		"video_path": "data/Korea_cctv/video/23_SUWON.mp4"
	},
	"34_SUWON": {
		"frame_num" : 6001, "movement_num": 11 , "index_from": 0, "index_to": 6000, "number_of_classes": 8,
		"video_path": "data/Korea_cctv/video/34_SUWON.mp4"
	}
}

# Set the main video information for run all
video_info = video_info_Korea_cctv_folders

segment_period = 10
base_factor	= 1.014405


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
		self.current_video_index   = -1
		self.video_name            = None
		self.is_run_all            = False
		self.is_run_one            = False

		# support for drawing UI
		self.last_draw_ts = 0.0

		uic.loadUi(ui_path, self)

		# Build layout
		# self.setWindowTitle(window_title)
		self.build_layout()
		self.reset_value()

	def build_layout(self):
		# Map logical button names to Qt widgets.
		# This makes enable/disable logic scalable when buttons are added/removed.
		self.button_widgets = {
			"start"  : self.button_start,
			"stop"   : self.button_stop,
			"video"  : self.button_video,
			"run_all": self.button_run_all,
			# add future buttons here only
		}

		# Build button
		self.button_start.clicked.connect(partial(self.button_released,TFE_BUTTON.START))
		self.button_stop.clicked.connect(partial(self.button_released,TFE_BUTTON.STOP))
		self.button_video.clicked.connect(partial(self.button_released, TFE_BUTTON.VIDEO))
		self.button_run_all.clicked.connect(partial(self.button_released, TFE_BUTTON.RUN_ALL))

	# MARK: Operation function

	@pyqtSlot()
	def open_video_dialog(self):
		# choice, ok = QInputDialog.getItem(
		# 	self,
		# 	"Select Input Type",
		# 	"Choose input:",
		# 	["File", "Folder"],
		# 	0,
		# 	False,
		# )
		# if not ok:
		# 	return
		#
		# selected_path = ""
		# if choice == "File":
		# 	selected_path, _ = QFileDialog.getOpenFileName(
		# 		self,
		# 		"Open Video/Image File",
		# 		"./data/Korea_cctv/video",
		# 		"Media Files (*.mp4 *.avi *.mov *.mkv *.png *.jpg *.jpeg);;All Files (*)",
		# 	)
		# else:
		selected_path = QFileDialog.getExistingDirectory(
			self,
			"Open Folder",
			"./data/Korea_cctv/video",
		)

		if not selected_path:
			return

		print(f"Process: {selected_path}")
		self.set_video_information(selected_path)

	def set_video_information(self, video_path):
		basename_noext = os.path.splitext(os.path.basename(video_path))[0]
		self.thread_video_args.dataset = os.path.basename(os.path.dirname(os.path.dirname(video_path)))
		self.thread_video_args.config  = os.path.join(os.path.dirname(os.path.dirname(video_path)), "configs", f"{basename_noext}.yaml")
		self.thread_video_args.output  = os.path.join(os.path.dirname(os.path.dirname(video_path)), "outputs", f"{basename_noext}.txt")
		self.thread_video_args.output_run_all = os.path.join(os.path.dirname(os.path.dirname(video_path)), "outputs", f"{self.thread_video_args.dataset}.txt")
		self.thread_video_args.groundtruth = os.path.join(os.path.dirname(os.path.dirname(video_path)), "groundtruths", f"{basename_noext}.txt")
		self.thread_video_args.visualize   = True
		self.thread_video_args.write_video = True
		self.video_name                    = basename_noext
		self.label_video_name.setText(f"Dataset : {self.thread_video_args.dataset} -- Video: {self.video_name}")

	def update_display_thread(self, create_camera, args):
		# store the function
		self.thread_video_function = create_camera
		# config=camera_hprams, visualize=args.visualize, write_video=args.write_video
		self.thread_video_args	   = deepcopy(args)

	def create_display_thread(
			self,
			CameraThread : Union[QThread, None] = None,
			args         : Union[dict, None]    = None
	):
		if isinstance(CameraThread, QThread):
			# create the video capture thread
			self.thread_video = CameraThread
			# connect its signal to the update_image slot
			self.thread_video.update_information.connect(self.update_information)
		elif callable(CameraThread):  # if this is a function create camera
			self.thread_video = CameraThread(args)
			# connect its signal to the update_image slot
			self.thread_video.update_information.connect(self.update_information)
			self.video_name  = os.path.splitext(os.path.basename(args.config))[0]
			self.label_video_name.setText(f"Dataset : {args.dataset} -- Video: {self.video_name}")

		# cleanup hooks once when creating the thread:
		self.thread_video.finished.connect(self.thread_video.deleteLater)

	def set_buttons_enabled(self, enabled_map: dict[str, bool]):
		# Apply state updates by logical button key.
		# Unknown keys are ignored so partial UI configs do not break runtime.
		for key, state in enabled_map.items():
			widget = self.button_widgets.get(key)
			if widget is not None:
				widget.setEnabled(state)

	def set_status(self, text: str, color: str):
		# Centralized status label styling for consistent UI state.
		self.label_status.setText(f"STATUS: {text}")
		self.label_status.setStyleSheet(f"color: {color}")

	def button_released(self, button_type: TFE_BUTTON):
		action_map = {
			TFE_BUTTON.START: {
				"status": ("START", "green"),
				"before": {"start": False, "stop": True, "video": False, "run_all": False},
				"call"  : lambda: (self.reset_value(), self.start_process()),
				"after" : {},
			},
			TFE_BUTTON.STOP: {
				"status": ("STOP", "red"),
				"before": {"start": False, "stop": False, "video": False, "run_all": False},
				"call"  : lambda: (self.reset_value(), self.stop_process(do_evaluation = False), self.reset_display()),
				"after" : {},
			},
			TFE_BUTTON.VIDEO: {
				"status": ("VIDEO", "white"),
				"before": {"start": True , "stop": True , "video": True , "run_all": True},
				"call"  : lambda: (self.stop_process(do_evaluation = False), self.open_video_dialog()),
				"after" : {},
			},
			TFE_BUTTON.CONFIG: {
				"status": ("CONFIG", "white"),
				"before": {"start": False, "stop": False, "video": False, "run_all": False},
				"call"  : self.stop_process,
				"after" : {"start": True},
			},
			TFE_BUTTON.RUN_ALL: {
				"status": ("RUN_ALL", "green"),
				"before": {"start": False, "stop": True , "video": False, "run_all": False},
				"call"  :lambda: (self.pre_run_all(), self.run_all()),
				"after" : {},
			},
		}

		action = action_map.get(button_type)
		if not action:
			return

		status_text, status_color = action["status"]
		self.set_status(status_text, status_color)
		self.set_buttons_enabled(action.get("before", {}))
		action["call"]()
		self.set_buttons_enabled(action.get("after", {}))

	def pre_run_all(self):
		# Set current video at start
		self.reset_value()
		self.current_video_index = 0
		self.is_run_all          = True

	def run_all(self):
		# NOTE: Fixed dataset list
		# list_video = [
		# 	"data/carla_weather/video/Town10HD_location_2.mp4",
		# 	"data/carla_weather/video/Town10HD_location_4.mp4",
		# 	"data/carla_weather/video/Town10HD_location_6.mp4",
		# 	"data/carla_weather/video/Town10HD_location_7.mp4",
		# ]
		if self.current_video_index > -1:
			# self.current_video_index show which video is running
			# self.set_video_information(list_video[self.current_video_index])
			self.set_video_information(list(video_info.values())[self.current_video_index]["video_path"])
			self.start_process()

	def start_process(self):
		self.is_run_one         = True
		self.process_start_time = timer()
		logger.info("Camera starting...")
		# start the thread
		if self.thread_video is not None:
			self.thread_video.start()
		elif self.thread_video_function is not None:
			self.create_display_thread(self.thread_video_function, self.thread_video_args)
			self.thread_video.start()
		logger.info("Camera running...")

	def stop_process(self, do_evaluation: bool = True):
		self.is_run_one = False
		logger.info("Camera stoping")
		# stop the thread
		thread = self.thread_video
		self.thread_video = None  # detach early to prevent re-entry issues

		if thread is not None:
			# Ask worker loop to stop (your thread already exposes stop()).
			if hasattr(thread, "stop"):
				thread.stop()

			# Ask Qt event loop in that thread to quit (safe even if not running exec()).
			thread.quit()

			# Wait briefly for clean shutdown.
			if not thread.wait(3000):
				logger.warning("Thread did not stop in time; forcing terminate()")
				thread.terminate()
				thread.wait(1000)

			# Schedule QObject deletion on owning thread/event loop.
			thread.deleteLater()

		self.process_end_time = timer()
		# Give time for the thread to finish
		logger.info("\nCamera stopped...")

		# evaluation
		if do_evaluation and self.video_name is not None:
			self.evaluate_one_result()

			# If the last video of run_all video finished
			if self.current_video_index == len(list(video_info.values())) - 1:
				self.evaluate_all_result()
				self.current_video_index = -1
				# self.button_run_all.setEnabled(True)
				self.is_run_all          = False

	# MARK: Update for display

	def update_information(self, information):
		""" Information send from the QThread Camera
		Args:
			information (dict):
		"""
		results    = information["result_count"]
		frame      = information["result_frame"]
		is_running = information["is_run"]

		# this is the place to annouce the camera stop
		if not is_running:
			self.label_evaluation.setText(f"Evaluation Result: {self.thread_video_args.output.replace(os.getcwd(), '')}")
			self.stop_process()  # stop the done process

			# check if run all
			# which help the program continue running after one video is done
			if self.is_run_all and self.current_video_index > -1:
				self.current_video_index += 1
				self.run_all()

			# check if dont run anything
			if not self.is_run_all and not self.is_run_one:
				self.reset_display()

		# update count result
		self.update_result_display(results)

		# update display
		self.update_main_display(frame)

	def update_result_display(self, results):
		if isinstance(results, str):
			self.label_result.setText(results)
		elif isinstance(results, list):
			results_str = ""
			for result in results:
				if "none" not in str.lower(result):
					results_str += f"{result}\n"
			self.label_result.setText(results_str)

	def update_main_display(self, frame):
		# NOTE: (optional) throttle UI painting to ~30 FPS while inference runs full speed:
		# now = timer()
		# if now - self.last_draw_ts < (1.0 / 30.0):
		# 	return
		# self.last_draw_ts = now

		# check whether frame is None or not
		if frame is None:
			return

		target_w = self.display_frame.width()
		target_h = self.display_frame.height()
		if target_w <= 0 or target_h <= 0:
			return

		# Resize only when needed.
		if frame.shape[1] != target_w or frame.shape[0] != target_h:
			# INTER_LINEAR is usually the best speed/quality tradeoff for live preview.
			cv_img = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
		else:
			cv_img = frame


		# Ensure memory layout is contiguous for QImage.
		if not cv_img.flags["C_CONTIGUOUS"]:
			cv_img = np.ascontiguousarray(cv_img)

		h, w, ch = cv_img.shape
		if ch != 3:
			# Guard: this renderer expects BGR888
			return

		bytes_per_line = ch * w
		qimg = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)

		# Keep copy() for safety (prevents dangling pointer after function returns).
		self.display_frame.setPixmap(QPixmap.fromImage(qimg.copy()))

	# MARK: Evaluation

	def reset_value(self):
		self.wRMSE_array          = []
		self.vehicleNum_array     = []
		self.period_process_array = []
		self.is_run_all           = False
		self.is_run_one           = False

	def reset_display(self):
		self.set_status("STOP", "red")
		for key, widget in self.button_widgets.items():
			widget.setEnabled(True)

	def evaluate_one_result(self):
		"""Evaluate the camera's performance.
		"""
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

		video_total_frame  = list(video_info.values())[self.current_video_index]["frame_num"]
		score_effetiveness = wRMSE / vehicle_num
		score_efficiency   = 1 - (total_process_time * base_factor) / (1.1 * float(video_total_frame))
		score_f1		   = 0.3 * score_efficiency + 0.7 * score_effetiveness

		print(f"vehicle_num	 : {vehicle_num}\n"
			  f"wRMSE		 : {wRMSE:.6f}\n"
			  f"nwRMSE	     : {wRMSE / vehicle_num:.6f}\n"
			  f"period	     : {total_process_time:.2f}\n"
			  f"index_video  : {self.current_video_index}\n"
			  f"effetiveness : {score_effetiveness:.6f}\n"
			  f"efficiency   : {score_efficiency:.6f}\n"
			  f"score_f1	 : {score_f1:.6f}\n"
			  )


		with open(self.thread_video_args.output, "a") as f_append:
			f_append.write(f"effetiveness : {score_effetiveness:.6f}\n"
						   f"efficiency   : {score_efficiency:.6f}\n"
						   f"score_f1	 : {score_f1:.6f}\n"
						   )

	def evaluate_all_result(self):
		video_total_frame  = sum([value["frame_num"] for value in video_info.values()])
		total_process_time = sum(self.period_process_array)

		score_effetiveness = sum(self.wRMSE_array) / sum(self.vehicleNum_array)
		score_efficiency   = 1 - (total_process_time * base_factor) / (1.1 * float(video_total_frame))
		score_f1		   = 0.3 * score_efficiency + 0.7 * score_effetiveness

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
						  f"score_f1	 : {score_f1:.6f}\n"
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
