# ==================================================================== #
# File name: file.py
# Author: Long H. Pham
# Date created: 03/29/2021
# ==================================================================== #
import os


# MARK: - Create


# MARK: - Read

def is_image_file(file: str) -> bool:
	""" Check if is a image file.
	
	Args:
		file (str):
			The file or path with extension.

	Returns:
		True or False.
	"""
	image_formats = [".bmp", ".jpg", ".jpeg", ".png", ".ppm", ".bmp"]  # Acceptable image suffixes
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in image_formats:
			return True
	return False


def is_json_file(file: str) -> bool:
	""" Check if is .json file.
	
	Args:
		file (str):
			The file or path with extension.

	Returns:
		True or False.
	"""
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in [".json"]:
			return True
	return False


def is_torch_saved_file(file: str) -> bool:
	""" Check if is a .pt or .pth file.
	
	Args:
		file (str):
			The file or path with extension.

	Returns:
		True or False.
	"""
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in [".pt", ".pth"]:
			return True
	return False


def is_engine_saved_file(file: str) -> bool:
	""" Check if is a .pt or .pth file.

	Args:
		file (str):
			The file or path with extension.

	Returns:
		True or False.
	"""
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in [".engine"]:
			return True
	return False


def is_txt_file(file: str) -> bool:
	""" Check if is .txt file.
	
	Args:
		file (str):
			The file or path with extension.
			
	Returns:
		True or False.
	"""
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in [".txt"]:
			return True
	return False


def is_video_file(file: str) -> bool:
	""" Check if is a video file.
	
	Args:
		file (str):
			The file or path with extension.

	Returns:
		True or False.
	"""
	video_formats = [".mov", ".avi", ".mp4", ".mpg", ".mpeg", ".m4v", ".wmv", ".mkv"]  # Acceptable video suffixes
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in video_formats:
			return True
	return False


def is_video_stream(stream: str) -> bool:
	"""Check if the given stream is of correct format.
	
	Args:
		stream (str):
	
	Returns:
		True or False.
	"""
	return "rtsp" in stream


def is_yaml_file(file: str) -> bool:
	""" Check if is a .yaml file.
	
	Args:
		file (str):
			The file or path with extension.

	Returns:
		True or False.
	"""
	if os.path.isfile(path=file):
		extension = os.path.splitext(file)[1]
		if extension in [".yaml", ".yml"]:
			return True
	return False


# MARK: - Update


# MARK: - Delete

