# ==================================================================== #
# File name: tracker.py
# Author: Long H. Pham and Duong N.-N. Tran
# Date created: 03/30/2021
#
# ``Tracker`` base class for all variant of tracker.
# It define a unify template to guarantee the input and output of all tracker are the same.
# Usually, each ``Tracker`` class is associate with a ``Track`` class
#
# Subclassing guide:
# 1. The package (i.e, the .py filename) should be in the template:
#    {tracker}_{track_motion_model}_{feature_used_to_track}
# ==================================================================== #
import abc
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from tss.road_objects import GMO
from tss.utils import printe


# MARK: - Tracker

class Tracker(metaclass=abc.ABCMeta):
	"""Tracker Base Class.
	
	Attributes:
		api (str):
			Name of the API that the tracker model is implemented from
		name (str):
			Name of the tracker method.
		max_age (int):
			The time to store the track before deleting,
				that mean track could live in max_age frame with no match bounding box, consecutive frame that track disappear
		min_hits (int):
			The number of frame which has matching bounding box of the detected object
				before the object is considered becoming the track
		iou_threshold (float):
			The Intersection over Union between two track with their bounding box
		frame_count (int):
			The current index of reading frame,
				The number of input frame with detection
		tracks (list):
			A list of ``Track``.
	"""
	
	# MARK: Magic Functions

	def __init__(
		self,
		name         : Optional[str] = None,
		max_age      : int   = 1,
		min_hits     : int   = 3,
		iou_threshold: float = 0.3,
		**kwargs
	):
		self.name              = name
		self.max_age           = max_age
		self.min_hits          = min_hits
		self.iou_threshold     = iou_threshold
		self.frame_count       = 0
		self.tracks: List[GMO] = []
	
	# MARK: Update
	
	def update(self, detections: Any):
		"""Update ``self.tracks`` with new detections.
		
		Args:
			detections (any):
				The newly detections.
				The type of detections depends on the tracked object.
				Also, it can be either a list or np.ndarray, depends on the coder.
		
		Requires:
			This method must be called once for each frame even with empty detections, just call update with empty container.
		"""
		printe("``update()`` has not been implemented yet")
		raise NotImplementedError
	
	def update_matched_tracks(
		self,
		matched   : Union[List, np.ndarray],
		detections: Any
	):
		"""Update the track that has been matched with new detection
		
		Args:
			matched (list or np.ndarray):
				Matching between self.tracks index and detection index.
			detections (any):
				The newly detections.
		"""
		printe("``update_matched_tracks()`` has not been implemented yet")
		raise NotImplementedError
	
	def create_new_tracks(
		self,
		unmatched_dets: Union[List, np.ndarray],
		detections    : Any
	):
		"""Create new tracks.
		
		Args:
			unmatched_dets (list or np.ndarray):
				Index of the newly detection in ``detections`` that has not matched with any tracks.
			detections (any):
				The newly detections.
		"""
		printe("``create_new_tracks()`` has not been implemented yet")
		raise NotImplementedError
	
	def delete_dead_tracks(self):
		"""Delete dead tracks.
		"""
		printe("``delete_dead_tracks()`` has not been implemented yet")
		raise NotImplementedError
	
	def associate_detections_to_tracks(self, dets: Any, trks: Any, **kwargs):
		"""Assigns detections to ``self.tracks``
		
		Args:
			dets (any):
				The newly detections.
				The type of detections depends on the specific tracker.
			trks (any):
				The current tracks.
				The type of tracks depends on the specific tracker.
	
		Returns:
			3 lists of matches, unmatched_detections and unmatched_trackers
		"""
		printe("``associate_detections_to_trackers()`` has not been implemented yet")
		raise NotImplementedError
