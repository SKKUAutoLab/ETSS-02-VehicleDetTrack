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
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import cv2
from munch import Munch

from tfe.road_objects import GMO
from tfe.ops import angle_between_arrays
from tfe.ops import AppleRGB
from tfe.ops import get_distance_function
from tfe.utils import data_dir
from tfe.utils import is_json_file
from tfe.utils import parse_config_from_json
from tfe.utils import printe


# MARK: - MOI (Movement of Interest)

class MOI(object):
	"""MOI (Movement of Interest)
	"""
	# MARK: Magic Functions

	def __init__(
		self,
		uuid              : Optional[int]                 = None,
		points            : Union[List[np.ndarray], None] = None,
		shape_type        : Optional[str]                 = None,
		offset            : Optional[int]                 = None,
		distance_function : str                           = "hausdorff",
		distance_threshold: float                         = 300.0,
		angle_threshold   : float                         = 45.0,
		color             : Tuple[int, int, int]          = None,
		**kwargs
	):
		super().__init__(**kwargs)
		self.uuid               = uuid
		self.points             = points
		self.shape_type         = shape_type
		self.offset             = offset
		self.distance_threshold = distance_threshold
		self.angle_threshold    = angle_threshold
		
		if uuid is None:
			self.color = AppleRGB.WHITE.value
		else:
			self.color = color if color else AppleRGB.values()[uuid]
		
		self.distance_function: Callable[[np.ndarray, np.ndarray], float] \
			= get_distance_function(name=distance_function)
		
		if self.points is None or not all(len(track) >= 2 for track in self.points):
			printe("Insufficient number of points in the moi's track.")
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
	def load_mois_from_file(
		cls,
		file   : str,
		dataset: Optional[str] = None,
		**kwargs
	):
		"""Load moi's points from external .json file.
		"""
		# NOTE: Get json file
		if dataset:
			path = os.path.join(data_dir, dataset, "rmois", file)
		else:
			path = os.path.join(data_dir, "rmois", file)
		if not is_json_file(file=path):
			printe(f"File not found or given a wrong file type at {path}.")
			raise FileNotFoundError
		
		# NOTE: Create moi road_objects
		data      = parse_config_from_json(json_path=path)
		data      = Munch.fromDict(d=data)
		mois_data = data.moi
		
		mois: List[MOI] = []
		for moi_data in mois_data:
			mois.append(cls(**moi_data, **kwargs))
		return mois
	
	# MARK: Matching
	
	@staticmethod
	def associate_moving_objects_to_mois(
		gmos      : List[GMO],
		mois                 ,
		shape_type: str = "linestrip"
	):
		"""A static method to check if a list of given moving objects belong to one of the MOIs in the image.
		"""
		if len(gmos) <= 0: return
		polygon_mois   = [m for m in mois if m.shape_type == "polygon"]
		linestrip_mois = [m for m in mois if m.shape_type == "linestrip"]
		
		if shape_type == "polygon":
			for o in gmos:
				if o.moi_uuid is None:
					o.moi_uuid = MOI.find_moi_for_bbox(bbox_xyxy=o.current_bbox, mois=polygon_mois)
		elif shape_type == "linestrip":
			for o in gmos:
				if o.moi_uuid is None:
					o.moi_uuid = MOI.best_matched_moi(object_track=o.trajectory, mois=linestrip_mois)[0]
		
	@staticmethod
	def find_moi_for_bbox(
		bbox_xyxy: np.ndarray,
		mois
	) -> Optional[int]:
		"""A static method to check if a given bbox belong to one of the many MOIs in the image.
		"""
		for moi in mois:
			if moi.is_center_in_or_touch_moi(bbox_xyxy=bbox_xyxy) > 0:
				return moi.uuid
		return None
	
	@staticmethod
	def best_matched_moi(
		object_track: np.ndarray,
		mois
	) -> Tuple[int, float]:
		"""Find the Moi that best matched with the object's track.
		"""
		# NOTE: Calculate distances between object track and all mois' tracks
		distances = []
		angles    = []
		for moi in mois:
			distances.append(moi.distances_with_track(object_track=object_track))
			angles.append(moi.angles_with_track(object_track=object_track))
		
		min_moi_uuid, min_distance = None, None
		for i, (d, a) in enumerate(zip(distances, angles)):
			if d is None or a is None:
				continue
			if (min_distance is not None) and (min_distance < d):
				continue
			min_distance = d
			min_moi_uuid = mois[i].uuid

		return min_moi_uuid, min_distance
	
	# MARK: Calculation
	
	def distances_with_track(self, object_track: np.ndarray) -> List[float]:
		"""Calculate the distance between object's track to the MOI's tracks.
		"""
		distance = self.distance_function(self.points, object_track)
		return None if (distance > self.distance_threshold) else distance
	
	def angles_with_track(self, object_track: np.ndarray) -> List[float]:
		"""Calculate the angle between object's track to the MOI's tracks
		"""
		angle = angle_between_arrays(self.points, object_track)
		return None if (angle > self.angle_threshold) else angle
	
	def is_center_in_or_touch_moi(self, bbox_xyxy: np.ndarray, compute_distance: bool = False) -> int:
		""" Check the bounding box touch MOI or not.
		"""
		c_x = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
		c_y = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
		return int(cv2.pointPolygonTest(self.points, (c_x, c_y), compute_distance))
	
	# MARK: Visualize
	
	def draw(self, drawing: np.ndarray):
		"""Draw the ROI.
		"""
		# NOTE: Draw MOI's direction
		pts = self.points.reshape((-1, 1, 2))
		if self.shape_type == "polygon":
			cv2.polylines(img=drawing, pts=[pts], isClosed=True, color=self.color, thickness=1, lineType=cv2.LINE_AA)
		elif self.shape_type == "linestrip":
			cv2.polylines(img=drawing, pts=[pts], isClosed=False, color=self.color, thickness=1, lineType=cv2.LINE_AA)
			cv2.arrowedLine(img=drawing, pt1=tuple(self.points[-2]), pt2=tuple(self.points[-1]), color=self.color, thickness=1, line_type=cv2.LINE_AA, tipLength=0.2)
			for i in range(len(self.points) - 1):
				cv2.circle(img=drawing, center=tuple(self.points[i]), radius=3, color=self.color, thickness=-1, lineType=cv2.LINE_AA)
				
		# NOTE: Draw MOI's id
		cv2.putText(img=drawing, text=f"{self.uuid}", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, org=tuple(self.points[-1]), color=self.color, thickness=2)
