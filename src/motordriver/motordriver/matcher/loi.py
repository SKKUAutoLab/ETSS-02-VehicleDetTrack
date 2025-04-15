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
from __future__ import annotations

import os
import sys
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import cv2
from munch import Munch

# NOTE: check import
try:
	from core.objects.gmo import GMO
except ImportError:
	pass

from core.utils.constants import AppleRGB
from core.io.filedir import is_json_file
from utils import (
	data_dir,
	parse_config_from_json,
)
from core.utils.rich import console, error_console


# MARK: - LOI (Lane of Interest)

class LOI(object):
	"""LOI (Lane of Interest)
	"""
	# MARK: Magic Functions

	def __init__(
		self,
		uuid              : Optional[int]                 = None,
		points            : Union[List[np.ndarray], None] = None,
		shape_type        : Optional[str]                 = None,
		offset            : Optional[int]                 = None,
		color             : Tuple[int, int, int]          = None,
		**kwargs
	):
		super().__init__(**kwargs)
		self.uuid               = uuid
		self.points             = points
		self.shape_type         = shape_type
		self.offset             = offset

		if uuid is None:
			self.color = AppleRGB.WHITE.value
		else:
			self.color = color if color else AppleRGB.values()[uuid]
		
		if self.points is None or not all(len(track) >= 2 for track in self.points):
			error_console.log("Insufficient number of points in the loi's track.")
			raise ValueError
	
	# MARK: Property
	
	@property
	def points(self) -> Union[np.ndarray, None]:
		return self._points
	
	@points.setter
	def points(self, points: Union[List, np.ndarray, None]):
		if isinstance(points, list):
			self._points = np.array(points, np.int32)
		elif isinstance(points, np.ndarray):
			self._points = points
	
	# MARK: Configure
	
	@classmethod
	def load_lois_from_file(
		cls,
		file   : str,
		dataset: Optional[str] = None,
		**kwargs
	):
		"""Load loi's points from external .json file.
		"""
		# NOTE: Get json file
		if dataset:
			path = os.path.join(data_dir, dataset, "rmois", file)
		else:
			path = os.path.join(data_dir, "rmois", file)
		if not is_json_file(path=path):
			error_console.log(f"File not found or given a wrong file type at {path}.")
			raise FileNotFoundError
		
		# NOTE: Create loi road_objects
		data      = parse_config_from_json(json_path=path)
		data      = Munch.fromDict(d=data)
		lois_data = data.loi
		
		lois: List[LOI] = []
		for loi_data in lois_data:
			lois.append(cls(**loi_data, **kwargs))
		return lois
	
	# MARK: Matching
	
	@staticmethod
	def associate_moving_objects_to_lois(
		gmos      : List[GMO],
		lois                 ,
		shape_type: str = "polygon"
	):
		"""A static method to check if a list of given moving objects belong to one of the LOIs in the image.
		"""
		if len(gmos) <= 0:
			return
		polygon_lois = [m for m in lois if m.shape_type == "polygon"]

		if shape_type == "polygon":
			for gmo in gmos:
				LOI.find_loi_for_bbox(gmo=gmo, lois=polygon_lois)

	@staticmethod
	def find_loi_for_bbox(
		gmo: GMO,
		lois
	) -> Optional[list, None]:
		"""A static method to check if a given bbox belong to one of the many LOIs in the image.
		"""
		for bbox_index in range(len(gmo.bboxes)):
			for loi in lois:
				if loi.is_center_in_or_touch_loi(bbox_xyxy=gmo.bboxes[bbox_index]) > 0:
					gmo.lanes_id[bbox_index] = loi.uuid

	# MARK: Calculation
	
	def is_center_in_or_touch_loi(
		self,
		bbox_xyxy: np.ndarray,
		compute_distance: bool = False
	) -> int:
		""" Check the bounding box touch LOI or not.
		"""
		c_x = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
		c_y = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
		return int(cv2.pointPolygonTest(self.points, (c_x, c_y), compute_distance))
	
	# MARK: Visualize
	
	def draw(self, drawing: np.ndarray):
		"""Draw the ROI.
		"""
		pass
