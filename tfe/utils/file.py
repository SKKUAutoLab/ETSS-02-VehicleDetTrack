# ==================================================================== #
# File name: file.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/29/2021
# ==================================================================== #
import os


# MARK: - Create


# MARK: - Read

def is_image_file(file: str) -> bool:
	""" Check if is a image file.
	"""
	image_formats = [".bmp", ".jpg", ".jpeg", ".png", ".ppm", ".bmp"]  # Acceptable image suffixes
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in image_formats:
			return True
	return False


def is_json_file(file: str) -> bool:
	""" Check if is .json file.
	"""
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in [".json"]:
			return True
	return False


def is_torch_saved_file(file: str) -> bool:
	""" Check if is a .pt or .pth file.
	"""
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in [".pt", ".pth"]:
			return True
	return False


def is_engine_saved_file(file: str) -> bool:
	""" Check if is a .pt or .pth file.
	"""
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in [".engine"]:
			return True
	return False


def is_txt_file(file: str) -> bool:
	""" Check if is .txt file.
	"""
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in [".txt"]:
			return True
	return False


def is_video_file(file: str) -> bool:
	""" Check if is a video file.
	"""
	video_formats = [".mov", ".avi", ".mp4", ".mpg", ".mpeg", ".m4v", ".wmv", ".mkv"]  # Acceptable video suffixes
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in video_formats:
			return True
	return False


def is_video_stream(stream: str) -> bool:
	"""Check if the given stream is of correct format.
	"""
	return "rtsp" in stream


def is_yaml_file(file: str) -> bool:
	""" Check if is a .yaml file.
	"""
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in [".yaml", ".yml"]:
			return True
	return False


# MARK: - Update


# MARK: - Delete

