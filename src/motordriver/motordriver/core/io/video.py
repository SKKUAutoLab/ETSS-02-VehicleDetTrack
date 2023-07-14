#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.io.filedir import create_dirs
from core.type.type import Arrays
from core.type.type import Dim3
from core.utils.image import is_channel_first
from core.utils.image import to_channel_last
from core.io.format import VideoFormat

__all__ = [
	"is_video_file",
	"is_video_stream",
	"VideoLoader",
	"VideoWriter"
]


# MARK: - Read

def is_video_file(path: Optional[str]) -> bool:
	"""Check if the given path is a video file."""
	if path is None:
		return False
	
	video_formats = VideoFormat.values()
	if os.path.isfile(path=path):
		extension = os.path.splitext(path.lower())[1]
		if extension in video_formats:
			return True
	return False


def is_video_stream(path: str) -> bool:
	"""Check if the given path is a video stream."""
	path = path.lower()
	return "rtsp" in path


# MARK: - VideoLoader/Writer

class VideoLoader:
	"""Video Loader loads frames from a video file or a video stream.

	Attributes:
		data (str):
			Data source. Can be a path to video file or a stream link.
		batch_size (int):
			Number of samples in one forward & backward pass.
		video_capture (VideoCapture):
			`VideoCapture` object from OpenCV.
		num_frames (int):
			Total number of frames in the video.
		index (int, optional):
			Current frame index.
	"""

	# MARK: Magic Functions

	def __init__(self, data: str, batch_size: int = 1):
		super().__init__()
		self.data          = data
		self.batch_size    = batch_size
		self.video_capture = None
		self.num_frames    = -1
		self.index         = 0

		self.init_video_capture(data=self.data)
		
	def __len__(self):
		"""Return the number of frames in the video.

		Returns:
			num_frames (int):
				>0 if the offline video.
				-1 if the online video.
		"""
		return self.num_frames  # number of frame, [>0 : video, -1 : online_stream]

	def __iter__(self):
		"""Returns an iterator starting at index 0."""
		self.index = 0
		return self

	def __next__(self):
		"""
		e.g.:
				>>> video_stream = VideoLoader("cam_1.mp4")
				>>> for image, index in enumerate(video_stream):

		Returns:
			images (np.ndarray):
				List of numpy.array images from OpenCV.
			indexes (list):
				List of image indexes in the video.
			files (list):
				List of image files.
			rel_paths (list):
				List of images' relative paths corresponding to data.
		"""
		if self.index >= self.num_frames:
			raise StopIteration
		else:
			images    = []
			indexes   = []
			files     = []
			rel_paths = []

			for i in range(self.batch_size):
				if self.index >= self.num_frames:
					break
					
				ret_val, image = self.video_capture.read()
				rel_path       = os.path.basename(self.data)
				
				images.append(image)
				indexes.append(self.index)
				files.append(self.data)
				rel_paths.append(rel_path)
				
				self.index += 1

			return images, indexes, files, rel_paths

	def __del__(self):
		"""Close the `video_capture` object."""
		self.close()

	# MARK: Configure
	
	def init_video_capture(self, data: str):
		"""Initialize `video_capture` object.
		
		Args:
			data (str):
				Data source. Can be a path to video file or a stream link.
		"""
		if is_video_file(data):
			self.video_capture = cv2.VideoCapture(data)
			self.num_frames    = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		elif is_video_stream(data):
			self.video_capture = cv2.VideoCapture(data)  # stream
			# Set buffer (batch) size
			self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, self.batch_size)
		
		if self.video_capture is None:
			raise IOError("Error when reading input stream or video file!")

	def close(self):
		"""Release the current `video_capture` object."""
		if self.video_capture:
			self.video_capture.release()


class VideoWriter:
	"""Video Writer saves images to a video file.

	Attributes:
		dst (str):
			Output video file.
		video_writer (VideoWriter):
			`VideoWriter` object from OpenCV.
		shape (tuple):
			Output size as [H, W, C]. This is also used to reshape the input.
		frame_rate (int):
			Frame rate of the video.
		fourcc (str):
			Fvideo codec. One of: ["mp4v", "xvid", "mjpg", "wmv1"].
		save_image (bool):
			Should write individual image?
		index (int):
			Current index.
	"""

	# MARK: Magic Functions

	def __init__(
		self,
		dst       : str,
		shape     : Dim3  = (480, 640, 3),
		frame_rate: float = 10,
		fourcc    : str   = "mp4v",
		save_image: bool  = False,
	):
		super().__init__()
		self.shape        = shape
		self.frame_rate   = frame_rate
		self.fourcc       = fourcc
		self.save_image	  = save_image
		self.video_writer = None
		self.index		  = 0

		self.init_video_writer(dst=dst)

	def __len__(self):
		"""Return the number of already written frames."""
		return self.index

	def __del__(self):
		"""Close the `video_writer` object."""
		self.close()

	# MARK: Configure
	
	def init_video_writer(self, dst: str):
		"""Initialize `video_writer` object.

		Args:
			dst (str):
				Output video file.
		"""
		if os.path.isdir(dst):
			parent_dir = dst
			self.dst   = os.path.join(parent_dir, f"result.mp4")
		else:
			parent_dir = str(Path(dst).parent)
			stem       = str(Path(dst).stem)
			self.dst   = os.path.join(parent_dir, f"{stem}.mp4")
		create_dirs(paths=[parent_dir])

		fourcc            = cv2.VideoWriter_fourcc(*self.fourcc)
		self.video_writer = cv2.VideoWriter(
			self.dst, fourcc, self.frame_rate,
			tuple([self.shape[1], self.shape[0]])  # Must be [W, H]
		)

		if self.video_writer is None:
			raise FileNotFoundError(f"Video file cannot be created at "
									f"{self.dst}.")

	def close(self):
		"""Release the `video_writer` object."""
		if self.video_writer:
			self.video_writer.release()

	# MARK: Write

	def write_frame(self, image: np.ndarray):
		"""Add a frame to video.

		Args:
			image (np.ndarray):
				Image for writing of shape [H, W, C].
		"""
		if is_channel_first(image):
			image = to_channel_last(image)

		if self.save_image:
			parent_dir = os.path.splitext(self.dst)[0]
			image_file = os.path.join(parent_dir, f"{self.index}.png")
			create_dirs(paths=[parent_dir])
			cv2.imwrite(image_file, image)

		self.video_writer.write(image)
		self.index += 1

	def write_frames(self, images: Arrays):
		"""Add batch of frames to video.

		Args:
			images (Arrays):
				Images.
		"""
		for image in images:
			self.write_frame(image=image)
