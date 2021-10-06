# ==================================================================== #
# File name: moi.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/28/2021
#
# Movement of Interest ``MOI`` defines the movement of the road used for matching road_objects movement.
# ==================================================================== #
# from __future__ import annotations

import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from cv2 import cv2
from munch import Munch

from tss.road_objects import GMO
from tss.ops import angle_between_arrays
from tss.ops import AppleRGB
from tss.ops import get_distance_function
from tss.utils import data_dir
from tss.utils import is_json_file
from tss.utils import parse_config_from_json
from tss.utils import printe


# MARK: - MOI (Movement of Interest)

class MOI(object):
	"""MOI (Movement of Interest)
	
	Attributes:
		uuid (int):
			The moi's unique id.
		tracks (list):
			List of tracks in the MOI.
		shape_type (string):
			The shape type.
		offset (int):
		
		distance_function (callable):
			The distance function
		distance_threshold (float):
			The maximum distance for counting with track
		angle_threshold (float):
			The maximum angle for counting with track
	"""
	# MARK: Magic Functions

	def __init__(
		self,
		uuid              : Optional[int]                 = None,
		tracks            : Union[List[np.ndarray], None] = None,
		offset            : Optional[int]                 = None,
		distance_function : str                           = "hausdorff",
		distance_threshold: float                         = 300.0,
		angle_threshold   : float                         = 45.0,
		color             : Tuple[int, int, int]          = None,
		**kwargs
	):
		super().__init__(**kwargs)
		self.uuid               = uuid
		self.tracks             = tracks
		self.offset             = offset
		self.distance_threshold = distance_threshold
		self.angle_threshold    = angle_threshold
		
		if uuid is None:
			self.color = AppleRGB.WHITE.value
		else:
			self.color = color if color else AppleRGB.values()[uuid]
		
		self.distance_function: Callable[[np.ndarray, np.ndarray], float] \
			= get_distance_function(name=distance_function)
		
		if self.tracks is None or not all(len(track) >= 2 for track in self.tracks):
			printe("Insufficient number of points in the moi's track.")
			raise ValueError
	
	# MARK: Property
	
	@property
	def tracks(self) -> Union[List[np.ndarray], None]:
		return self._tracks
	
	@tracks.setter
	def tracks(self, tracks: Union[List[np.ndarray], None]):
		if isinstance(tracks, list):
			self._tracks = [np.array(track, np.int32) for track in tracks]
	
	# MARK: Configure
	
	@classmethod
	def load_mois_from_file(
		cls,
		file   : str,
		dataset: Optional[str] = None,
		**kwargs
	) :
		"""Load moi's points from external .json file.
		
		Args:
			file (str):
				Give the roi file. Example a path "..data/aicity2021/roi_data/cam_n.json", so provides ``cam_n.json``.
			dataset (str, optional):
				The name of the dataset to work on.
		
		Returns:
			mois (list):
				Return the list of Moi road_objects.
		"""
		# TODO: Get json file
		if dataset:
			path = os.path.join(data_dir, dataset, "rmois", file)
		else:
			path = os.path.join(data_dir, "rmois", file)
		if not is_json_file(file=path):
			printe(f"File not found or given a wrong file type at {path}.")
			raise FileNotFoundError
		
		# TODO: Create moi road_objects
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
		gmos: List[GMO],
		mois
	):
		"""A static method to check if a list of given moving objects belong to one of the MOIs in the image.

		Args:
			gmos (list):
				The list of moving object.
			mois (list):
				The list of MOIs in the image.
		"""
		for o in gmos:
			o.moi_uuid = MOI.best_matched_moi(object_track=o.trajectory, mois=mois)[0]
	
	@staticmethod
	def associate_moving_object_to_mois(
		gmo : GMO,
		mois
	):
		"""A static method to check if a given moving object belong to one of the MOIs in the image.

		Args:
			gmo (GMO):
				The moving object.
			mois (list):
				The list of MOIs in the image.
		"""
		gmo.moi_uuid = MOI.best_matched_moi(object_track=gmo.trajectory, mois=mois)[0]
		
	@staticmethod
	def best_matched_moi(
		object_track: np.ndarray,
		mois
	) -> Tuple[int, float]:
		"""Find the Moi that best matched with the object's track.
		
		Args:
			object_track (np.ndarray):
				The object's track as an array of points.
			mois (list):
				List of MOI.

		Returns:
			(id, min_dist):
				The best match moi's id and min distance.
		"""
		# TODO: Calculate distances between object track and all mois' tracks
		distances_dict = Munch()
		for moi in mois:
			distances_dict[moi.uuid]           = Munch()
			distances_dict[moi.uuid].distances = moi.distances_with_track(object_track=object_track)
			distances_dict[moi.uuid].angles    = moi.angles_with_track(object_track=object_track)
		
		# TODO: Return the moi's id that has min distance and angle to object track
		final_min_id, final_min_dist = None, None
		for (id, value) in distances_dict.items():
			distances = [d for (d, a) in zip(value.distances, value.angles) if d and a]
			min_dist  = min(distances) if len(distances) > 0 else None
			
			if min_dist is None:
				continue
			elif (final_min_dist is not None) and (min_dist > final_min_dist):
				continue
				
			final_min_dist = min_dist
			final_min_id   = id

		return final_min_id, final_min_dist
	
	# MARK: Calculation
	
	def distances_with_track(self, object_track: np.ndarray) -> List[float]:
		"""Calculate the distance between object's track to the MOI's tracks.
		
		If distance > self.distance_threshold, return None.
		
		Args:
			object_track (np.ndarray):
				The object's trajectory as an array of points.
				
		Returns:
			distances (np.ndarray):
				distance values between object's track with each track array.
		"""
		distances = [self.distance_function(track, object_track) for track in self.tracks]
		for i in range(len(distances)):
			if distances[i] > self.distance_threshold:
				distances[i] = None
		return distances
	
	def angles_with_track(self, object_track: np.ndarray) -> List[float]:
		"""Calculate the angle between object's track to the MOI's tracks
		
		If angle > self.angle_threshold, return None.
		
		Args:
			object_track (np.ndarray):
				The object's trajectory as an array of points.
				
		Returns:
			angles (np.ndarray):
				angle values between object's track with each track array.
		"""
		angles = [angle_between_arrays(track, object_track) for track in self.tracks]
		for i in range(len(angles)):
				if angles[i] > self.angle_threshold:
					angles[i] = None
		return angles
	
	# MARK: Visualize
	
	def draw(self, drawing: np.ndarray):
		"""Draw the ROI.
		
		Args:
			drawing (np.ndarray):
				The drawing canvas.
		
		Returns:
			drawing (np.ndarray):
				The drawing canvas.
		"""
		for track in self.tracks:
			# TODO: Draw MOI's direction
			pts = track.reshape((-1, 1, 2))
			cv2.polylines(img=drawing, pts=[pts], isClosed=False, color=self.color, thickness=1, lineType=cv2.LINE_AA)
			cv2.arrowedLine(img=drawing, pt1=tuple(track[-2]), pt2=tuple(track[-1]), color=self.color, thickness=1, line_type=cv2.LINE_AA, tipLength=0.2)
			for i in range(len(track) - 1):
				cv2.circle(img=drawing, center=tuple(track[i]), radius=3, color=self.color, thickness=-1, lineType=cv2.LINE_AA)
				
			# TODO: Draw MOI's id
			cv2.putText(img=drawing, text=f"{self.uuid}", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, org=tuple(track[-1]), color=self.color, thickness=2)
			
		return drawing
