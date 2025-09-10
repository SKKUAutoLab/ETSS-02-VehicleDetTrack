import sys
import os
from functools import partial
from typing import Optional

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
		self.setStyleSheet("background-color:#a4cfda")

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
	change_pixmap_signal = pyqtSignal(np.ndarray)

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
				self.change_pixmap_signal.emit(frame)

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
		self.display_width   = width
		self.display_height = height

		# Build layout
		self.frame = QLabel()
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
		cv_img = cv2.resize(cv_img, (self.display_width, self.display_height))
		convert = QImage(cv_img, cv_img.shape[1], cv_img.shape[0], cv_img.strides[0],
		                 QImage.Format.Format_BGR888)
		return QPixmap.fromImage(convert)


class MainWindow(QMainWindow):

	# MARK: Magic Functions

	def __init__(
		self,
		window_title = "Traffic Flow Estimation",
		*args, **kwargs
	):
		super(MainWindow, self).__init__()
		# Build layout
		self.setWindowTitle(window_title)
		self.build_layout()

		self.create_video_thread()

	def build_layout(self):
		# Build main window
		self.setFixedSize(1280, 800)

		# Build button
		self.button_start = QPushButton("START")
		self.button_start.released.connect(partial(self.button_released,TFE_BUTTON.START))
		self.button_stop = QPushButton("STOP")
		self.button_stop.released.connect(partial(self.button_released,TFE_BUTTON.STOP))
		self.button_video = QPushButton("VIDEO")
		self.button_video.released.connect(partial(self.button_released, TFE_BUTTON.VIDEO))
		self.button_config = QPushButton("CONFIG")
		self.button_config.released.connect(partial(self.button_released, TFE_BUTTON.CONFIG))

		# Build label
		self.label_video_name = QLabel()
		self.label_status     = QLabel()
		self.label_result     = LabelResult()
		self.video_show       = DisplayImageWidget(1080, 720)
		self.label_video_name.setStyleSheet("background-color:#ed7560")
		self.label_status.setStyleSheet("background-color:#b8dbf0")


		# Add to layout
		self.layout = QGridLayout()
		self.layout.addWidget(self.video_show, 1, 0, 3, 5)
		self.layout.addWidget(self.label_video_name, 0, 0, 1, 1)
		self.layout.addWidget(self.label_status, 4, 5, 1, 1)
		self.layout.addWidget(self.label_result, 1, 5, 3, 1)
		self.layout.addWidget(self.button_video, 4, 0, 1, 1)
		self.layout.addWidget(self.button_config, 4, 1, 1, 1)
		self.layout.addWidget(self.button_start, 4, 2, 1, 1)
		self.layout.addWidget(self.button_stop, 4, 3, 1, 1)

		# Build container for layout
		self.container = QWidget()
		self.container.setLayout(self.layout)

		# Set the central widget of the Window.
		self.setCentralWidget(self.container)

	def create_video_thread(self):
		# create the video capture thread
		self.thread_video = VideoThread()
		# connect its signal to the update_image slot
		self.thread_video.change_pixmap_signal.connect(self.video_show.update_image)

	# MARK: Operation function

	def button_released(self, button_type: TFE_BUTTON):
		if button_type is TFE_BUTTON.START:
			self.label_status.setText("STATUS: START")
			self.label_status.setStyleSheet('color: green')
			self.start_process()

		elif button_type is TFE_BUTTON.STOP:
			self.label_status.setText("STATUS: STOP")
			self.label_status.setStyleSheet('color: red')
			self.stop_process()

		elif button_type is TFE_BUTTON.VIDEO:
			self.label_status.setText("STATUS: VIDEO")
			self.label_status.setStyleSheet('color: white')
			self.stop_process()

		elif button_type is TFE_BUTTON.CONFIG:
			self.label_status.setText("STATUS: CONFIG")
			self.label_status.setStyleSheet('color: white')
			self.stop_process()

	def start_process(self):
		# start the thread
		self.thread_video.start()

	def stop_process(self):
		# stop the thread
		self.thread_video.terminate()


def main():
	# Create an instance of QApplication
	# QApplication manages the GUI application's control flow and main settings.
	# sys.argv is a list in Python, which contains the command-line arguments passed to the script.
	app = QApplication(sys.argv)
	# print(QStyleFactory.keys())
	app.setStyle('Windows')

	# Create an instance of our MainWindow class
	window = MainWindow()

	# Show the window on the screen
	window.show()

	# Start the application's event loop
	# The exec() method enters the main loop of the application and waits until exit() is called
	app.exec()


if __name__ == "__main__":
	main()
