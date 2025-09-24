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

import abc
import uuid
from timeit import default_timer as timer
from typing import Dict, Callable
from typing import Optional
from typing import Tuple
from uuid import UUID

import cv2
import numpy as np

from tfe.objects.instance import Instance
from tfe.utils.bbox import bbox_xyxy_center
from tfe.utils.distance import get_distance_function
# from tfe.utils.point import distance_between_points
from tfe.utils.label import get_majority_label


# MARK: - GeneralObject

class GeneralObject(metaclass=abc.ABCMeta):

	
	# MARK: Class Property
	
	min_travelled_distance: float = 20.0
	
	# MARK: Magic Functions

	def __init__(
		self,
		id         : UUID                 = uuid.uuid4(),
		frame_index: Optional[int]        = None,
		timestamp  : float                = timer(),
		bbox       : Optional[np.ndarray] = None,
		clip_bbox  : Optional[np.ndarray] = None,
		polygon    : Optional[np.ndarray] = None,
		confidence : float                = 0.0,
		label      : Optional[Dict]       = None,
		distance_function : str           = "euclidean",
		**kwargs
	):
		super().__init__(**kwargs)
		self.id            = id
		self.frame_indexes = [frame_index] if (frame_index is not None) else []
		self.timestamps    = [timestamp]   if (timestamp is not None)   else []
		self.bboxes        = [bbox]        if (bbox is not None)        else []
		self.polygons      = [polygon]     if (polygon is not None)     else []
		self._trajectory   = np.array([self.current_bbox_center]) if (bbox is not None) else np.empty((0, 2))
		self.confidence    = confidence
		self.labels        = [label]       if (label is not None)       else []
		self.distance_function: Callable[[np.ndarray, np.ndarray], float] \
			= get_distance_function(name=distance_function)

	# MARK: Property
	
	@property
	def last_frame_index(self) -> np.ndarray:
		return self.frame_indexes[-1]
	
	@property
	def last_timestamp(self) -> float:
		return self.timestamps[-1]
	
	@property
	def current_bbox(self) -> np.ndarray:
		return self.bboxes[-1]
	
	@property
	def current_bbox_center(self) -> np.ndarray:
		return bbox_xyxy_center(self.bboxes[-1])
	
	@property
	def current_polygon(self) -> np.ndarray:
		return self.polygons[-1]
	
	@property
	def trajectory(self) -> np.ndarray:
		return self._trajectory
	
	@property
	def travelled_distance(self) -> np.ndarray:
		# return distance_between_points(self._trajectory[0], self._trajectory[-1])
		return self.distance_function(self._trajectory[0], self._trajectory[-1])

	@property
	def current_label(self) -> Dict:
		return self.labels[-1]
	
	@property
	def label_by_majority(self) -> Dict:
		return get_majority_label(object_labels=self.labels)
	
	@property
	def label_id_by_majority(self) -> int:
		return self.label_by_majority.id
	
	# MARK: Configure
	
	@classmethod
	def go_from_detection(cls, instance: Instance, **kwargs):

		return cls(
			frame_index = instance.frame_index,
			timestamp   = instance.timestamp,
			bbox        = instance.bbox,
			polygon     = instance.polygon,
			confidence  = instance.confidence,
			label       = instance.label,
			roi_uuid    = instance.roi_uuid,
			**kwargs
		)
	
	# MARK: Update
	
	def update_go(
		self,
		frame_index: int,
		bbox       : np.ndarray,
		confidence : float,
		label      : Dict,
		polygon    : Optional[np.ndarray] = None,  # Later used for instance segmentation model
		timestamp  : float                = timer(),
		**kwargs
	):
		self.frame_indexes.append(frame_index)
		self.timestamps.append(timestamp)
		self.bboxes.append(bbox)
		self.labels.append(label)
		self.confidence = confidence
		if polygon:
			self.polygons = np.append(self.polygons, [polygon], axis=0)
			
		# if distance_between_points(self.trajectory[-1], self.current_bbox_center) >= GeneralObject.min_travelled_distance:
		if self.distance_function(self.trajectory[-1], self.current_bbox_center) >= GeneralObject.min_travelled_distance:
			self._trajectory = np.append(self._trajectory, [self.current_bbox_center], axis=0)
	
	def update_go_from_detection(self, instance: Instance, **kwargs):
		self.update_go(
			frame_index = instance.frame_index,
			timestamp   = instance.timestamp,
			bbox        = instance.bbox,
			confidence  = instance.confidence,
			label       = instance.label,
			polygon     = instance.polygon
		)
		
	# MARK: Visualize
	
	def draw(
		self,
		drawing   : np.ndarray,
		bbox      : bool = True,
		clip_bbox : bool = False,
		polygon   : bool = False,
		label     : bool = True,
		trajectory: bool = False,
		color     : Optional[Tuple[int, int, int]] = None
	):

		color = color if color is not None else self.label_by_majority.color

		if bbox:
			curr_bbox = self.current_bbox
			if False:
				# NOTE: only ellipse for the object
				width = abs(curr_bbox[2] - curr_bbox[0])
				height = abs(curr_bbox[3] - curr_bbox[1])
				drawing = cv2.ellipse(drawing, center=tuple(self.current_bbox_center), axes=(width // 8, height // 8),
									  angle=0, startAngle=0, endAngle=360, color=color, thickness=-1)
			else:
				# NOTE: bounding box cover the object
				cv2.rectangle(img=drawing,
				              pt1=(int(curr_bbox[0]), int(curr_bbox[1])),
				              pt2=(int(curr_bbox[2]), int(curr_bbox[3])),
							  color=color, thickness=2)

				cv2.circle(img=drawing, center=tuple(np.array(self.current_bbox_center, dtype=int)), radius=3, thickness=-1, color=color)

		if polygon:
			curr_polygon = self.current_polygon
			pts          = curr_polygon.reshape((-1, 1, 2))
			cv2.polylines(img=drawing, pts=pts, isClosed=True, color=color, thickness=2)

		if label:
			bbox_tl    = self.current_bbox[0:2]
			curr_label = self.label_by_majority
			font       = cv2.FONT_HERSHEY_SIMPLEX
			org        = (bbox_tl[0] + 5, bbox_tl[1])
			cv2.putText(img=drawing, text=curr_label.name, fontFace=font, fontScale=1.0, org=np.array(org, dtype=int), color=color, thickness=2)
		
		if trajectory:
			pts = self.trajectory.reshape((-1, 1, 2))
			cv2.polylines(img=drawing, pts=[np.array(pts, dtype=int)], isClosed=False, color=color, thickness=2)
			for point in self.trajectory:
				cv2.circle(img=drawing, center=np.array(point, dtype=int), radius=3, thickness=2, color=color)
