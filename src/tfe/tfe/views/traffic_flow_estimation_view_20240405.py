import sys
import os
import time
from copy import deepcopy
from functools import partial
from typing import Optional, Union

import numpy as np
import cv2
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *

class TFE_BUTTON:
	"""
	This class represents the different classes of objects that can be detected by the AI system.
	Each class is represented by an integer value.
	"""
	START    : int = 1
	STOP     : int = 2
	VIDEO    : int = 3
	CONFIG   : int = 4


class LabelResult(QLabel):

	# MARK: Magic Functions

	def __init__(
		self,
		height: Optional[int] = 720
	):
		super(LabelResult, self).__init__()

		self.setFixedHeight(height)
		self.setStyleSheet("font-size: 18pt;")
		self.setAlignment(Qt.AlignmentFlag.AlignTop)


	# MARK: Operation function

	def update_result(self, results):
		if isinstance(results, str):
			self.setText(results)
		elif isinstance(results, list):
			results_str = ""
			for result in results:
				results_str += f"{result}\n"
			self.setText(results_str)


class VideoThread(QThread):

	# MARK: Magic Functions
	updateFrame = pyqtSignal(np.ndarray)
	updateResult = pyqtSignal(list)

	def __init__(self):
		super().__init__()
		self.run_flag = True

	# MARK: Operation function

	def run(self):
		# DEBUG: Test run video
		self.test_run_video()
		pass

	def stop(self):
		"""Sets run flag to False and waits for thread to finish"""
		self.run_flag = False
		self.wait()

	def test_run_video(self):
		cap = cv2.VideoCapture(
			'/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/data/carla_rain/video/Town10HD_location_2.mp4')

		# Check if camera opened successfully
		if not cap.isOpened():
			print("Error opening video stream or file")

		# Read until video is completed
		while cap.isOpened():
			# Capture frame-by-frame
			ret, frame = cap.read()
			# if not ret:
			# 	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			# 	continue

			if ret:
				# Display the resulting frame
				self.updateFrame.emit(frame)

				# Frame per second show on display
				self.msleep(30)

		cap.release()


class DisplayImageWidget(QWidget):

	# MARK: Magic Functions

	def __init__(
		self,
		width : Optional[int] = 1080,
		height: Optional[int] = 720
	):
		super(DisplayImageWidget, self).__init__()

		# Set size
		self.setFixedWidth(width)
		self.setFixedHeight(height)
		self.disply_width   = width
		self.display_height = height

		# Build layout
		self.frame  = QLabel()
		self.layout = QHBoxLayout(self)
		self.layout.addWidget(self.frame)
		self.frame.setStyleSheet("background-color:#ff77aa")

	# MARK: Operation function

	@pyqtSlot(np.ndarray)
	def update_image(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		self.frame.setPixmap(qt_img)

	def convert_cv_qt(self, cv_img):
		cv_img  = cv2.resize(cv_img, (self.disply_width, self.display_height))
		convert = QImage(cv_img, cv_img.shape[1], cv_img.shape[0], cv_img.strides[0],
		                 QImage.Format.Format_BGR888)
		return QPixmap.fromImage(convert)


class TFEMainWindow(QMainWindow):

	# MARK: Magic Functions

	def __init__(
			self,
			window_title = "Traffic Flow Estimation",
			*args, **kwargs
	):
		super(TFEMainWindow, self).__init__()
		# Inite
		self.thread_video          = None
		self.thread_video_function = None

		# Build layout
		self.setWindowTitle(window_title)
		self.build_layout()

	def build_layout(self):
		# Build main window
		self.setFixedSize(1280, 800)

		# Build button
		self.button_start = QPushButton("START")
		self.button_start.setEnabled(False)
		self.button_start.released.connect(partial(self.button_released,TFE_BUTTON.START))
		self.button_stop = QPushButton("STOP")
		self.button_stop.setEnabled(False)
		self.button_stop.released.connect(partial(self.button_released,TFE_BUTTON.STOP))
		self.button_video = QPushButton("VIDEO")
		self.button_video.released.connect(partial(self.button_released, TFE_BUTTON.VIDEO))
		# self.button_config = QPushButton("CONFIG")
		# self.button_config.released.connect(partial(self.button_released, TFE_BUTTON.CONFIG))

		# Build label
		self.label_video_name = QLabel()
		self.label_status     = QLabel()
		self.label_result     = LabelResult()
		self.display_frame    = DisplayImageWidget(1080, 720)
		self.label_video_name.setStyleSheet("background-color:#ed7560")
		self.label_status.setStyleSheet("background-color:#b8dbf0")


		# Add to layout
		self.layout = QGridLayout()
		self.layout.addWidget(self.display_frame, 1, 0, 3, 5)
		self.layout.addWidget(self.label_video_name, 0, 0, 1, 2)
		self.layout.addWidget(self.label_status, 4, 3, 1, 1)
		self.layout.addWidget(self.label_result, 1, 5, 3, 1)
		self.layout.addWidget(self.button_video, 4, 0, 1, 1)
		# self.layout.addWidget(self.button_config, 4, 1, 1, 1)
		self.layout.addWidget(self.button_start, 4, 1, 1, 1)
		self.layout.addWidget(self.button_stop, 4, 2, 1, 1)

		# Build container for layout
		self.container = QWidget()
		self.container.setLayout(self.layout)

		# Set the central widget of the Window.
		self.setCentralWidget(self.container)

	# MARK: Operation function

	@pyqtSlot()
	def open_video_dialog(self):
		video_path = QFileDialog.getOpenFileName(
			self,
			"Open Video File",
			"./data",
			"All Files (*);; Python Files (*.py);; PNG Files (*.png)",
		)

		print(f"Process video: {video_path[0]}")
		basename_noext = os.path.splitext(os.path.basename(video_path[0]))[0]
		self.thread_video_args.config      = os.path.join(os.path.dirname(os.path.dirname(video_path[0])), "configs", f"{basename_noext}.yaml")
		self.thread_video_args.dataset     = os.path.basename(os.path.dirname(os.path.dirname(video_path[0])))
		self.thread_video_args.visualize   = True
		self.thread_video_args.write_video = False


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
		if CameraThread is None:
			# create the video capture thread
			self.thread_video = VideoThread()
			# connect its signal to the update_image slot
			self.thread_video.updateFrame.connect(self.display_frame.update_image)
			self.thread_video.updateResult.connect(self.label_result.update_result)
			self.label_video_name.setText("DEFAULT")
		elif isinstance(CameraThread, QThread):
			# create the video capture thread
			self.thread_video = CameraThread
			# connect its signal to the update_image slot
			self.thread_video.updateFrame.connect(self.display_frame.update_image)
			self.thread_video.updateResult.connect(self.label_result.update_result)
		elif callable(CameraThread):  # if this is a function create camera
			self.thread_video = CameraThread(args)
			# connect its signal to the update_image slot
			self.thread_video.updateFrame.connect(self.display_frame.update_image)
			self.thread_video.updateResult.connect(self.label_result.update_result)
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
			self.button_stop.setEnabled(True)

		elif button_type is TFE_BUTTON.STOP:
			self.label_status.setText("STATUS: STOP")
			self.label_status.setStyleSheet('color: red')
			self.button_start.setEnabled(False)
			self.button_stop.setEnabled(False)
			self.button_video.setEnabled(False)
			self.stop_process()
			self.button_start.setEnabled(True)
			self.button_video.setEnabled(True)

		elif button_type is TFE_BUTTON.VIDEO:
			self.label_status.setText("STATUS: VIDEO")
			self.label_status.setStyleSheet('color: white')
			self.button_start.setEnabled(False)
			self.button_stop.setEnabled(False)
			self.button_video.setEnabled(False)
			self.stop_process()
			self.open_video_dialog()
			self.button_video.setEnabled(True)
			self.button_start.setEnabled(True)

		elif button_type is TFE_BUTTON.CONFIG:
			self.label_status.setText("STATUS: CONFIG")
			self.label_status.setStyleSheet('color: white')
			self.button_start.setEnabled(False)
			self.button_stop.setEnabled(False)
			self.button_video.setEnabled(False)
			self.stop_process()
			self.button_start.setEnabled(True)

	def start_process(self):
		print("Camera starting...")
		# start the thread
		if self.thread_video is not None:
			self.thread_video.start()
		elif self.thread_video_function is not None:
			self.create_display_thread(self.thread_video_function, self.thread_video_args)
			self.thread_video.start()
		print("Camera running...")

	def stop_process(self):
		# stop the thread
		if self.thread_video is not None:
			self.thread_video.terminate()
		self.thread_video = None
		# Give time for the thread to finish
		time.sleep(2)
		print("\nCamera stopped...")


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
