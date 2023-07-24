#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class to store persistent objects
"""

from __future__ import annotations

import uuid
from collections import Counter
from timeit import default_timer as timer
from typing import Optional
from typing import Union

import cv2
import numpy as np

from core.data.class_label import majority_voting
from core.type.type import Color
from .instance import Instance

__all__ = [
	"BaseObject"
]


# MARK: - BaseObject

class BaseObject:
	"""Base class for all objects.
	
	Requires:
		It is required to be subclassed with the motion model. In case you want
		to use it without tracking or counting functions.
	
	Attributes:
		id_ (int, str):
			Object unique ID.
		instances (list):
			List of all instances of this object.
		timestamp (float):
			Time when the object is created.
	"""

	# MARK: Magic Functions

	def __init__(
		self,
		id      : Union[int, str]                 = uuid.uuid4().int,
		instance : Optional[Union[list, Instance]] = None,
		timestamp: float 					   	   = timer(),
		*args, **kwargs
	):
		"""

		Args:
			id (int, str):
				Object unique ID.
			instance (list, Instance, optional):
				List of all instances of this object. Default: `None`.
			timestamp (float):
				Time when the object is created.
		"""
		super().__init__()
		self.id        = id
		self.timestamp = timestamp
		self.instances : list[Instance] = []

		if isinstance(instance, Instance):
			self.instances = [instance]
		else:
			self.instances = (instance if (instance is not None) else [])

	# MARK: Properties

	@property
	def class_labels(self) -> list:
		"""Get the list of all class_labels of the object."""
		return [instance.class_label for instance in self.instances]

	@property
	def current_bbox(self) -> np.ndarray:
		"""Get the latest det_bbox of the object."""
		return self.instances[-1].bbox
	
	@property
	def current_bbox_center(self) -> np.ndarray:
		"""Get the latest center of the object."""
		return self.instances[-1].bbox_center

	@property
	def current_confidence(self) -> float:
		"""Get the latest confidence score."""
		return self.instances[-1].confidence

	@property
	def current_class_label(self) -> dict:
		"""Get the latest label of the object."""
		return self.instances[-1].class_label

	@property
	def current_frame_index(self) -> int:
		"""Get the latest frame index of the object."""
		return self.instances[-1].frame_index

	@property
	def current_instance(self) -> Instance:
		"""Get the latest instance of the object."""
		return self.instances[-1]

	@property
	def current_polygon(self) -> np.ndarray:
		"""Get the latest polygon of the object."""
		return self.instances[-1].polygon

	@property
	def current_roi_id(self) -> Union[int, str]:
		"""Get the latest ROI's id of the object."""
		return self.instances[-1].roi_id

	@property
	def current_timestamp(self) -> float:
		"""Get the last time the object has been updated."""
		return self.instances[-1].timestamp

	@property
	def label_by_majority(self) -> dict:
		"""Get the major class_label of the object."""
		return majority_voting(self.class_labels)
	
	@property
	def label_id_by_majority(self) -> int:
		"""Get the most popular label's id of the object."""
		return self.label_by_majority["id"]

	@property
	def roi_id_by_majority(self) -> Union[int, str]:
		"""Get the major ROI's id of the object."""
		roi_id = Counter(self.roi_ids).most_common(1)
		return roi_id[0][0]

	@property
	def roi_ids(self) -> list[Union[int, str]]:
		"""Get the list ROI's ids of the object."""
		return [instance.roi_id for instance in self.instances]

	# MARK: Update
	
	def update(self, instance: Optional[Instance], **kwargs):
		"""Update with new instance.
		
		Args:
			instance (Instance, optional):
				Instance of the object.
		"""
		self.instances.append(instance)

	# MARK: Visualize
	
	def draw(
		self,
		drawing   : np.ndarray,
		bbox      : bool            = True,
		polygon   : bool            = False,
		label     : bool            = True,
		color     : Optional[Color] = None
	) -> np.ndarray:
		"""Draw the object into the `drawing`.
		
		Args:
			drawing (np.ndarray):
				Drawing canvas.
			bbox (bool):
				Should draw the detected bbox? Default: `True`.
			polygon (bool):
				Should draw polygon? Default: `False`.
			label (bool):
				Should draw label? Default: `True`.
			color (tuple):
				Primary color. Default: `None`.
		"""
		color = (color if (color is not None)
				 else self.label_by_majority["color"])
		bbox  = self.current_bbox

		if bbox is not None:
			cv2.rectangle(
				img       = drawing,
				pt1       = (bbox[0], bbox[1]),
				pt2       = (bbox[2], bbox[3]),
				color     = color,
				thickness = 2
			)
			bbox_center = self.current_bbox_center.astype(int)
			cv2.circle(
				img       = drawing,
				center    = tuple(bbox_center),
				radius    = 3,
				thickness = -1,
				color     = color
			)
		
		"""
		if polygon is not None:
			pts = self.current_polygon.reshape((-1, 1, 2))
			cv2.polylines(
				img=drawing, pts=pts, isClosed=True, color=color, thickness=2
			)
		"""
		
		if label is not None:
			bbox_tl    = bbox[0:2]
			curr_label = self.label_by_majority
			font       = cv2.FONT_HERSHEY_SIMPLEX
			org        = (bbox_tl[0] + 5, bbox_tl[1])
			cv2.putText(
				img       = drawing,
				text      = curr_label["name"],
				fontFace  = font,
				fontScale = 1.0,
				org       = org,
				color     = color,
				thickness = 2,
			)
