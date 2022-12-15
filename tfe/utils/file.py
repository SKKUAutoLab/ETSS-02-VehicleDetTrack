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
#
# ``Tracker`` base class for all variant of tracker.
# It define a unify template to guarantee the input and output of all tracker are the same.
# Usually, each ``Tracker`` class is associate with a ``Track`` class
#
# Subclassing guide:
# 1. The package (i.e, the .py filename) should be in the template:
#    {tracker}_{track_motion_model}_{feature_used_to_track}
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

