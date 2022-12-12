# ==================================================================== #
# File name: sort_kalman_bbox.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/30/2021
# ==================================================================== #
# from __future__ import annotations

from typing import List
from typing import Union

import numpy as np
from filterpy.kalman import KalmanFilter

from tfe.detector import Detection
from tfe.road_objects import GMO
from tfe.tracker import Tracker
from tfe.ops import bbox_xyxy_to_z
from tfe.ops import iou_batch
from tfe.ops import x_to_bbox_xyxy

np.random.seed(0)


# MARK: - Track

class KalmanBBoxTrack(GMO):
	"""Kalman Bounding Box Track
	"""

	# MARK: Magic Functions

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		# TODO: Define Kalman Filter (constant velocity model)
		self.kf   = KalmanFilter(dim_x=7, dim_z=4)
		self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
		self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
		
		self.kf.R[2:, 2:] *= 10.0
		self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
		self.kf.P         *= 10.0
		self.kf.Q[-1, -1] *= 0.01
		self.kf.Q[4:, 4:] *= 0.01
		
		# Here we assume that the ``GMO`` object has already been init().
		# So ``self.current_bbox`` return the first bbox value.
		self.kf.x[:4] = bbox_xyxy_to_z(self.current_bbox)
	
	@classmethod
	def track_from_detection(cls, detection: Detection, **kwargs):
		"""Create ``GMO`` object from ``Detection`` object.
		
		Args:
			detection (Detection):
		
		Returns:
			gmo (GMO):
				The GMO object.
		"""
		return cls(
			frame_index = detection.frame_index,
			timestamp   = detection.timestamp,
			bbox        = detection.bbox,
			polygon     = detection.polygon,
			confidence  = detection.confidence,
			label       = detection.label,
			roi_uuid    = detection.roi_uuid,
			**kwargs
		)
	
	# MARK: Property
	
	@property
	def matching_features(self):
		"""Return the features used to matched tracked objects with new detections.
		"""
		return self.current_bbox
		
	# MARK: Motion Model
	
	def update_motion_state(self, **kwargs):
		"""Updates the state of the motion model with observed bbox.
		"""
		self.time_since_update = 0
		self.history           = []
		self.hits             += 1
		self.hit_streak       += 1
		self.kf.update(bbox_xyxy_to_z(self.matching_features))
		
	def predict_motion_state(self):
		"""Advances the state of the motion model and returns the predicted estimate.
		"""
		if (self.kf.x[6] + self.kf.x[2]) <= 0:
			self.kf.x[6] *= 0.0
		self.kf.predict()
		self.age += 1
		if self.time_since_update > 0:
			self.hit_streak = 0
		self.time_since_update += 1
		self.history.append(x_to_bbox_xyxy(self.kf.x))
		return self.history[-1]

	def current_motion_state(self):
		"""
		Returns the current motion model estimate.
		"""
		return x_to_bbox_xyxy(self.kf.x)
	

# MARK: - Tracker

class Sort(Tracker):
	"""Sort (Simple Online Realtime Tracker)
	
	Attributes:
		Same as ``Tracker``
	"""
	# MARK: Magic Functions
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.tracks: List[KalmanBBoxTrack] = []
		
	# MARK: Update
	
	def update(self, detections: List[Detection]):
		"""Update ``self.tracks`` with new detections.
		
		Args:
			detections (list):
				The list of newly ``Detection`` objects.
		
		Requires:
			This method must be called once for each frame even with empty detections, just call update with empty list [].

		Returns:
		
		"""
		self.frame_count += 1  # Should be the same with VideoReader.frame_idx
		
		# TODO: Extract and convert bbox from detections for easier use.
		if len(detections) > 0:
			# dets - a numpy array of detections in the format [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score],...]
			dets = np.array([np.append(np.float64(d.bbox), np.float64(d.confidence)) for d in detections])
		else:
			dets = np.empty((0, 5))
		
		# TODO: Get predicted locations from existing trackers.
		trks   = np.zeros((len(self.tracks), 5))
		to_del = []
		for t, trk in enumerate(trks):
			pos    = self.tracks[t].predict_motion_state()[0]
			trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
			if np.any(np.isnan(pos)):
				to_del.append(t)
		trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
		
		# TODO: Find 3 lists of matches, unmatched_detections and unmatched_trackers
		for t in reversed(to_del):
			self.tracks.pop(t)
		matched, unmatched_dets, unmatched_trks = self.associate_detections_to_tracks(dets, trks)
		
		# TODO: Update matched trackers with assigned detections
		self.update_matched_tracks(matched=matched, detections=detections)
			
		# TODO: Create and initialise new trackers for unmatched detections
		self.create_new_tracks(unmatched_dets=unmatched_dets, detections=detections)
		
		# TODO: Remove dead tracklets
		self.delete_dead_tracks()
	
	def update_matched_tracks(
		self,
		matched   : Union[List, np.ndarray],
		detections: List[Detection]
	):
		"""Update the track that has been matched with new detection
		
		Args:
			matched (list or np.ndarray):
				Matching between self.tracks index and detection index.
			detections (any):
				The newly detections.
		"""
		for m in matched:
			track_idx     = m[1]
			detection_idx = m[0]
			# HERE, we call ``GMO.update_gmo()``. This contains all necessary functions to update the whole GMO object.
			self.tracks[track_idx].update_gmo(detections[detection_idx])
			
			# IF you don't call the function above, then call the following functions:
			# self.tracks[track_idx].update_go_from_detection(detection=detections[detection_idx])
			# self.tracks[track_idx].update_motion_state()
	
	def create_new_tracks(
		self,
		unmatched_dets: Union[List, np.ndarray],
		detections    : List[Detection]
	):
		"""Create new tracks.
		
		Args:
			unmatched_dets (list or np.ndarray):
				Index of the newly detection in ``detections`` that has not matched with any tracks.
			detections (any):
				The newly detections.
		"""
		for i in unmatched_dets:
			new_trk = KalmanBBoxTrack.track_from_detection(detections[i])
			self.tracks.append(new_trk)
	
	def delete_dead_tracks(self):
		"""Delete dead tracks.
		"""
		i = len(self.tracks)
		for trk in reversed(self.tracks):
			d = trk.current_motion_state()[0]  # Get the current bounding box of Kalman Filter
			#if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
				# ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
			i -= 1
			# TODO: Remove dead tracklets
			if trk.time_since_update > self.max_age:
				self.tracks.pop(i)
		
	def associate_detections_to_tracks(
		self,
		dets: np.ndarray,
		trks: np.ndarray,
		**kwargs
	):
		"""Assigns detections to ``self.tracks``
		
		Args:
			dets (np.ndarray):
				The list of newly ``Detection`` objects.
			trks (np.ndarray):

		Returns:
			3 lists of matches, unmatched_detections and unmatched_trackers
		"""
		if len(trks) == 0:
			return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty((0, 5), dtype=int)
		
		iou_matrix = iou_batch(dets, trks)
		
		if min(iou_matrix.shape) > 0:
			a = (iou_matrix > self.iou_threshold).astype(np.int32)
			if a.sum(1).max() == 1 and a.sum(0).max() == 1:
				matched_indices = np.stack(np.where(a), axis=1)
			else:
				matched_indices = linear_assignment(-iou_matrix)
		else:
			matched_indices = np.empty(shape=(0, 2))
		
		unmatched_detections = []
		for d, det in enumerate(dets):
			if d not in matched_indices[:, 0]:
				unmatched_detections.append(d)
		unmatched_trackers = []
		for t, trk in enumerate(trks):
			if t not in matched_indices[:, 1]:
				unmatched_trackers.append(t)
		
		# filter out matched with low IOU
		matches = []
		for m in matched_indices:
			if iou_matrix[m[0], m[1]] < self.iou_threshold:
				unmatched_detections.append(m[0])
				unmatched_trackers.append(m[1])
			else:
				matches.append(m.reshape(1, 2))
				
		if len(matches) == 0:
			matches = np.empty((0, 2), dtype=int)
		else:
			matches = np.concatenate(matches, axis=0)
		
		return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# MARK: - Utils

def linear_assignment(cost_matrix):
	"""

	Args:
		cost_matrix:

	Returns:

	"""
	try:
		import lap
		_, x, y = lap.lapjv(cost_matrix, extend_cost=True)
		return np.array([[y[i], i] for i in x if i >= 0])  #
	except ImportError:
		from scipy.optimize import linear_sum_assignment
		x, y = linear_sum_assignment(cost_matrix)
		return np.array(list(zip(x, y)))
