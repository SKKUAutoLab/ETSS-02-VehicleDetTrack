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
# ==================================================================== #
import glob
import os
from typing import List
from typing import Optional

import cv2
import numpy as np

from tfe.ops import image_channel_last
from tfe.ops import is_channel_last
from tfe.utils import is_image_file
from tfe.utils import printe


# MARK: - ImageReader

class ImageReader(object):
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		filename  : Optional[str] = None,
		dirname   : Optional[str] = None,
		batch_size: Optional[int] = 4,
		**kwargs
	):
		super().__init__(**kwargs)
		if dirname is not None:
			self.images = [img for img in glob.glob(os.path.join(dirname, "*")) if is_image_file(img)]
		elif filename is not None and is_image_file(os.path.join(dirname, filename)):
			self.images = [filename]
		else:
			printe("Error when reading input image files")
			raise IOError
		
		self.number_of_frame = len(self.images)
		self.batch_size      = batch_size
		
	def __iter__(self):
		""" The returns an iterator from them.
		"""
		self.frame_idx = 0
		return self
	
	def __next__(self):
		""" The next iterator for capture video
		"""
		images    = []
		frame_ids = []
		for _ in range(self.batch_size):
			if self.frame_idx < self.number_of_frame:
				self.frame_idx += 1
				frame_ids.append(self.frame_idx)
				images.append(cv2.imread(self.images[self.frame_idx]))
				
		return images, frame_ids
	
	def __len__(self):
		return self.number_of_frame  # number of images


# MARK: - ImageWriter

class ImageWriter(object):

	# MARK: Magic Functions
	
	def __init__(
		self,
		output_dir : str,
		extension  : Optional[str] = ".jpg",
		**kwargs
	):
		super().__init__(**kwargs)
		self.output_dir = output_dir
		self.extension  = extension
		self.image_name = output_dir.replace("\\", "/").split("/")[-1]
		self.frame_idx  = 0
		
	def __len__(self):
		return self.frame_idx  # number of already written images
		
	# NOTE: Setup stream
	
	def write_frame(self, images: List[np.ndarray]):
		""" Add batch of frames to folder.
		"""
		try:
			for image in images:
				# DEBUG:
				# print(image.shape)

				if is_channel_last(image):
					cv2.imwrite(image, os.path.join(self.output_dir, f"{self.image_name}{self.extension}"))
				else:
					cv2.imwrite(image_channel_last(image), os.path.join(self.output_dir, f"{self.image_name}{self.extension}"))
				self.frame_idx += 1
			return True
		except:
			printe(f"Something happen while writing image {self.output_dir}/{self.image_name}{self.extension}")
			return False
