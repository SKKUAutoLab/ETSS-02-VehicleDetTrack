#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class to store results from detectors. Attributes includes:
bounding box, confident score, class, uuid, ...
"""

from __future__ import annotations

import uuid
from timeit import default_timer as timer
from typing import Optional
from typing import Union

import cv2
import numpy as np

from thermal_pedestrian.core.type.type import Color
from thermal_pedestrian.core.utils.bbox import bbox_xyxy_center
from thermal_pedestrian.core.utils.bbox import bbox_xyxy_to_cxcyrh

__all__ = [
	"Instance"
]


# MARK: - Instance

class Instance:
	"""Instance Dataclass. Convert raw detected output from detector to easy
	to use namespace.
	
	Attributes:
		id_ (int, str):
			Object unique ID.
		roi_id (int, str, optional):
			Unique ID of the ROI that the object is in. Else `None`.
			Default: `None`.
		bbox (np.ndarray, optional):
			Bounding box points as
			[top_left x, top_left y, bottom_right x, bottom_right y].
			Default: `None`.
		polygon (np.ndarray, optional):
			List of points. Default: `None`.
		confidence (float, optional):
			Confidence score. Default: `None`.
		class_label (dict, optional):
			Classlabel dict. Default: `None`.
		frame_index (int):
			Index of frame when the Detection is created. Default: `None`.
		timestamp (float):
			Time when the object is created.
	"""
	
	# MARK: Magic Functions
	# SUGAR: co add them may attribute
	def __init__(
		self,
		id         : Union[int, str, list]      = uuid.uuid4().int,
		roi_uuid   : Union[int, str, None]      = None,
		bbox       : Optional[np.ndarray]       = None,
		polygon    : Optional[np.ndarray]       = None,
		confidence : Optional[float]            = None,
		class_label: Optional[dict]             = None,
		frame_index: Optional[int]              = None,
		class_id   : Optional[int]              = None,
		image_size : Optional[list, np.ndarray] = None,
		video_name : Optional[str]              = None,
		timestamp  : float                      = timer(),
		feature                                 = None,
		image                                   = None,
		*args, **kwargs
	):
		super().__init__()
		self.roi_uuid    = roi_uuid
		self.label       = class_label

		self.id          = id
		self.bbox        = bbox  # [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
		self.polygon     = polygon
		self.confidence  = confidence
		self.class_label = class_label
		self.frame_index = frame_index
		self.class_id    = class_id
		self.image_size  = image_size
		self.video_name  = video_name
		self.timestamp   = timestamp
		self.feature     = feature
		self.image       = image
	
	# MARK: Properties
	
	@property
	def bbox_cxcyrh(self):
		"""Return the bbox as
		[center_x, center_y, ratio, height]."""
		return bbox_xyxy_to_cxcyrh(self.bbox)

	@property
	def bbox_center(self):
		"""Return the bbox's center."""
		return bbox_xyxy_center(self.bbox)
	
	@property
	def bbox_tl(self):
		"""Return the bbox's top left corner."""
		return self.bbox[0:2]
	
	# MARK: Visualize
	
	def draw(
		self,
		drawing: np.ndarray,
		bbox   : bool            = False,
		polygon: bool            = False,
		label  : bool            = True,
		score  : bool            = False,
		color  : Optional[Color] = None
	) -> np.ndarray:
		"""Draw the road_objects into the `drawing`.
		
		Args:
			drawing (np.ndarray):
				Drawing canvas.
			bbox (bool):
				Should draw the detected bbox? Default: `False`.
			polygon (bool):
				Should draw polygon? Default: `False`.
			label (bool):
				Should draw label? Default: `True`.
			color (tuple):
				Primary color. Default: `None`.
		"""
		color = color if (color is not None) else self.class_label["color"]
		
		if bbox:
			cv2.rectangle(
				img=drawing, pt1=(self.bbox[0], self.bbox[1]),
				pt2=(self.bbox[2], self.bbox[3]), color=color, thickness=2
			)
		
		if polygon:
			pts = self.polygon.reshape((-1, 1, 2))
			cv2.polylines(
				img=drawing, pts=pts, isClosed=True, color=color, thickness=2
			)
		
		if label:
			if score:
				label_text = f"{self.class_label['name']}-{self.confidence:.2f}"
			else:
				label_text = self.class_label["name"]

			font = cv2.FONT_HERSHEY_SIMPLEX
			org  = (self.bbox_tl[0] + 5, self.bbox_tl[1])
			cv2.putText(
				img=drawing, text=label_text, fontFace=font,
				fontScale=1.0, org=org, color=color, thickness=2
			)
