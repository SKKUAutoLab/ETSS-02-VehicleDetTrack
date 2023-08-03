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
from core.objects.gmo import GMO
from core.utils.rich import console, error_console


# MARK: - Tracker

class Tracker(metaclass=abc.ABCMeta):
	"""Tracker Base Class.
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
		error_console.log("``update()`` has not been implemented yet")
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
		error_console.log("``update_matched_tracks()`` has not been implemented yet")
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
		error_console.log("``create_new_tracks()`` has not been implemented yet")
		raise NotImplementedError
	
	def delete_dead_tracks(self):
		"""Delete dead tracks.
		"""
		error_console.log("``delete_dead_tracks()`` has not been implemented yet")
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
		error_console.log("``associate_detections_to_trackers()`` has not been implemented yet")
		raise NotImplementedError
