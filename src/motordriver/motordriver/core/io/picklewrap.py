#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import multiprocessing
import os
from glob import glob
from pathlib import Path
from typing import Optional
from typing import Union

import cv2
import numpy as np
import pickle
# import pyvips
import torch
import torchvision
from joblib import delayed
from joblib import Parallel
from multipledispatch import dispatch
from PIL import ExifTags
from PIL import Image
from torch import Tensor

from core.io.filedir import create_dirs
from core.type.type import Arrays
from core.enhance.normalize import denormalize_naive
from core.utils.backend import interpolation_vision_backend_from_int
from core.utils.backend import interpolation_vision_backend_from_str
from core.utils.backend import VisionBackend
from core.utils.image import is_channel_first
from core.utils.image import to_channel_last
from core.io.format import ImageFormat

__all__ = [
	"PickleLoader",
]


# MARK: - Validate


# MARK: - PickleLoader/Writer

class PickleLoader:
	"""Image Loader retrieves and loads image(s) from a filepath, a pathname
	pattern, or directory.

	Attributes:
		data (str):
			Data source. Can be a path to an image file or a directory.
			It can be a pathname pattern to pickles.
		batch_size (int):
			Number of samples in one forward & backward pass.
		pickle_files (list):
			List of image files found in the data source.
		num_pickles (int):
			Total number of pickles.
		index (int):
			Current index.
	"""

	# MARK: Magic Functions

	def __init__(self, data: str, batch_size: int = 1):
		super().__init__()
		self.data        = data
		self.batch_size  = batch_size
		self.pickle_file = []
		self.num_elements  = -1
		self.index       = 0
		
		self.init_pickle_files(data=self.data)

	def __len__(self):
		"""Return the number of pickles in the `pickle_files`."""
		return self.num_elements  # Number of pickles
	
	def __iter__(self):
		"""Return an iterator starting at index 0."""
		self.index = 0
		return self

	def __next__(self):
		"""Next iterator.
		
		Examples:
			>>> video_stream = PickleLoader("cam_1.mp4")
			>>> for index, image in enumerate(video_stream):
		
		Returns:
			pickles (np.ndarray):
				List of image file from opencv with `np.ndarray` type.
			indexes (list):
				List of image indexes.
			files (list):
				List of image files.
			rel_paths (list):
				List of pickles' relative paths corresponding to data.
		"""
		if self.index >= self.num_elements:
			raise StopIteration
		else:
			elements   = []
			indexes    = []

			for i in range(self.batch_size):
				if self.index >= self.num_elements:
					break
				
				element = self.pickle_file[self.index]

				elements.append(element)
				indexes.append(self.index)

				self.index += 1

			return elements, indexes
	
	# MARK: Configure
	
	def init_pickle_files(self, data: str):
		"""Initialize list of image files in data source.
		
		Args:
			data (str):
				Data source. Can be a path to an image file or a directory.
				It can be a pathname pattern to pickles.
		"""
		self.pickle_file = pickle.load(open(data, 'rb'))
		self.num_elements = len(self.pickle_file)

	def list_pickle_files(self, data: str):
		"""Alias of `init_pickle_files()`."""
		self.init_pickle_files(data=data)



