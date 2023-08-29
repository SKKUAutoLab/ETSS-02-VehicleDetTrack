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
# from __future__ import annotations

import os
from typing import List
from typing import Optional
from typing import Union

import cv2
import numpy as np
from munch import Munch

from core.objects.instance import Instance
from core.utils.constants import AppleRGB as colors
from core.io.filedir import is_json_file
from utils import (
	data_dir,
	parse_config_from_json,
)
from core.utils.rich import error_console

# MARK: - ROI (Region of Interest)

class ROI(object):
	"""ROI (Region of Interest)
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		uuid      : Optional[int]        = None,
		points    : Optional[np.ndarray] = None,
		shape_type: Optional[str]        = None,
		**kwargs
	):
		super().__init__(**kwargs)
		self.uuid       = uuid
		self.shape_type = shape_type
		self.points     = points

		# NOTE: Load ROI points from file
		if self.points is None or len(self.points) < 2:
			error_console.log("Insufficient number of points in the roi.")
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
	def load_rois_from_file(
		cls,
		file   : str,
		dataset: Optional[str] = None,
		**kwargs
	) :
		"""Load roi from external .json file.
		"""
		# NOTE: Get json file
		if dataset:
			path = os.path.join(data_dir, dataset, "rmois", file)
		else:
			path = os.path.join(data_dir, "rmois", file)
		if not is_json_file(path=path):
			error_console.log(f"File not found or given a wrong file type at {path}.")
			raise FileNotFoundError
		
		# NOTE: Create moi road_objects
		data       = parse_config_from_json(json_path=path)
		data       = Munch.fromDict(d=data)
		rois_data  = data.roi
		
		rois: List = []
		for roi_data in rois_data:
			rois.append(cls(**roi_data, **kwargs))
		return rois
	
	# MARK: Validate
	
	@staticmethod
	def associate_detections_to_rois(
		detections: List[Instance],
		rois
	):
		"""A static method to check if a given bbox belong to one of the many rois in the image.
		"""
		for d in detections:
			d.roi_uuid = ROI.find_roi_for_bbox(bbox_xyxy=d.bbox, rois=rois)
	
	@staticmethod
	def find_roi_for_bbox(
		bbox_xyxy: np.ndarray,
		rois
	) -> Optional[int]:
		"""A static method to check if a given bbox belong to one of the many ROIs in the image.
		"""
		for roi in rois:
			if roi.is_center_in_or_touch_roi(bbox_xyxy=bbox_xyxy, compute_distance=True) >= -50:
				return roi.uuid
		return None
	
	def is_bbox_in_or_touch_roi(self, bbox_xyxy: np.ndarray, compute_distance: bool = False) -> int:
		""" Check the bounding box touch ROI or not
		"""
		# DEBUG:
		# print(type(self.points))
		# print(self.points)
		# print((bbox_xyxy[0], bbox_xyxy[1]))
		# Convert from [n, 2] to [n, 1, 2], new format of opencv
		# points = np.array([[point] for point in self.points])
		# print(type(points))
		# print(points)
		# print((bbox_xyxy[0], bbox_xyxy[1]))

		tl = cv2.pointPolygonTest(self.points, (int(bbox_xyxy[0]), int(bbox_xyxy[1])), compute_distance)
		tr = cv2.pointPolygonTest(self.points, (int(bbox_xyxy[2]), int(bbox_xyxy[1])), compute_distance)
		br = cv2.pointPolygonTest(self.points, (int(bbox_xyxy[2]), int(bbox_xyxy[3])), compute_distance)
		bl = cv2.pointPolygonTest(self.points, (int(bbox_xyxy[0]), int(bbox_xyxy[3])), compute_distance)
		
		if tl > 0 and tr > 0 and br > 0 and bl > 0:
			return 1
		elif tl < 0 and tr < 0 and br < 0 and bl < 0:
			return min(tl, tr, br, bl)
		else:
			return 0
	
	def is_center_in_or_touch_roi(self, bbox_xyxy: np.ndarray, compute_distance: bool = False) -> int:
		""" Check the bounding box touch ROI or not.
		"""
		c_x = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
		c_y = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
		return int(cv2.pointPolygonTest(self.points, (c_x, c_y), compute_distance))
	
	# MARK: Visualize

	def draw(self, drawing: np.ndarray):
		"""Draw the ROI.
		"""
		color = colors.GREEN.value
		pts   = self.points.reshape((-1, 1, 2))
		cv2.polylines(img=drawing, pts=[pts], isClosed=True, color=color, thickness=2)
		return drawing
