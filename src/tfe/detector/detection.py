# ==================================================================== #
# File name: detection.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/27/2021
# ==================================================================== #
import uuid
from timeit import default_timer as timer
from typing import Dict
from typing import Optional
from uuid import UUID

import cv2
import numpy as np
from typing import Tuple

from tfe.ops import bbox_xyah
from tfe.ops import bbox_xyxy_center


# MARK: - Detection

class Detection(object):
	"""Detection Data Class.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		id         : UUID                 = uuid.uuid4(),
		frame_index: Optional[int]        = None,
		timestamp  : float                = timer(),
		bbox       : Optional[np.ndarray] = None,
		polygon    : Optional[np.ndarray] = None,
		confidence : float                = 0.0,
		label      : Optional[Dict]       = None,
		roi_uuid   : Optional[int]        = None,
		**kwargs
	):
		super().__init__(**kwargs)
		self.id          = id
		self.frame_index = frame_index
		self.timestamp   = timestamp
		self.bbox        = bbox
		self.polygon     = polygon
		self.confidence  = confidence
		self.label       = label
		self.roi_uuid    = roi_uuid
	
	# MARK: Property
	
	@property
	def bbox_xyah(self):
		return bbox_xyah(bbox_xyxy=self.bbox)

	@property
	def bbox_center(self):
		return bbox_xyxy_center(bbox_xyxy=self.bbox)
	
	@property
	def bbox_tl(self):
		return self.bbox[0:2]
	
	# MARK: Visualize
	
	def draw(
		self,
		drawing   : np.ndarray,
		bbox      : bool = False,
		polygon   : bool = False,
		label     : bool = True,
		color     : Optional[Tuple[int, int, int]] = None
	) -> np.ndarray:
		"""Draw the road_objects into the ``drawing``.
		"""
		color = color if color is not None else self.label.color
		
		if bbox:
			curr_bbox = self.bbox
			cv2.rectangle(img=drawing, pt1=(curr_bbox[0], curr_bbox[1]), pt2=(curr_bbox[2], curr_bbox[3]), color=color, thickness=2)
		
		if polygon:
			pts = self.polygon.reshape((-1, 1, 2))
			cv2.polylines(img=drawing, pts=pts, isClosed=True, color=color, thickness=2)
		
		if label:
			curr_label = self.label
			font       = cv2.FONT_HERSHEY_SIMPLEX
			org        = (self.bbox_tl[0] + 5, self.bbox_tl[1])
			cv2.putText(img=drawing, text=curr_label.name, fontFace=font, fontScale=1.0, org=org, color=color, thickness=2)
