# ==================================================================== #
# File name: gmo.py
# Author: Long H. Pham and Duong N.-N. Tran
# Date created: 03/31/2021
# ==================================================================== #
# from __future__ import annotations

from typing import List

from tss.camera.roi import ROI
from tss.detector import Detection
from tss.ops import AppleRGB
from tss.road_objects.general_object import GeneralObject
from tss.road_objects.motion_model import MotionModel
from tss.road_objects.moving_model import MovingModel
from tss.road_objects.moving_model import MovingState


# MARK: - GMO (General Moving Object)

class GMO(GeneralObject, MotionModel, MovingModel):
	"""GMO (General Moving Object) = GeneralObject + MotionModel + CountingModel
	
	This class includes update functions to the object.
	
	Requires:
		This is an abstract class. It is required to be subclassed.
	
	Attributes:
		- Same attributes as ``GeneralObject``
		- Same attributes as ``MotionModel``
		- Same attributes as ``CountingModel``
		
		state (State):
			The current state of the road_objects.
	"""
	
	# MARK: Class Property
	
	min_entering_distance: int = 0    # Min distance when an object enters the ROI to be Confirmed.
	min_traveled_distance: int = 100  # Min distance between first trajectory point with last trajectory point.
	min_hit_streak       : int = 10   # Min number of "consecutive"'" frame has that track appear.
	max_age              : int = 1    # Max frame to wait until a dead track can be counted.
	
	# MARK: Magic Functions
	
	def __init__(self, **kwargs):
		GeneralObject.__init__(self, **kwargs)
		MotionModel.__init__(self, **kwargs)
		MovingModel.__init__(self, **kwargs)
		
	# MARK: Configure
	
	@classmethod
	def gmo_from_detection(cls, detection: Detection, **kwargs):
		"""Create ``GMO`` object from ``Detection`` object.
		
		Args:
			detection (Detection):
		
		Returns:
			gmo (GMO):
				The GMO object.
		"""
		return cls(
			frame_index = detection.frame_index,
			bbox        = detection.bbox,
			polygon     = detection.polygon,
			confidence  = detection.confidence,
			label       = detection.label,
			roi_uuid    = detection.roi_uuid,
			**kwargs
		)
		
	# MARK: Update
	
	def update_gmo(self, detection: Detection):
		"""Update the whole GMO object with newly matched ``Detection``.

		Args:
			detection (Detection):
				The newly matched ``Detection`` object.
		"""
		# TODO: First update ``GeneralObject``
		self.update_go_from_detection(detection=detection)
		
		# TODO: Second, update motion model
		self.update_motion_state()
		
	def update_moving_state(self, rois: List[ROI], **kwargs):
		"""Update the current state of the road_objects.
		
		The state diagram is as follow:
		
				(exist >= 10 frames)  (road_objects cross counting line)   (after being counted
				(in roi)                                                    by counter)
		_____________          _____________                  ____________        ___________        ________
		| Candidate | -------> | Confirmed | ---------------> | Counting | -----> | Counted | -----> | Exit |
		-------------          -------------                  ------------        -----------        --------
			  |                       |                                                                  ^
			  |_______________________|__________________________________________________________________|
								(mark by tracker when road_objects's max age > threshold)
		"""
		roi = next((roi for roi in rois if roi.uuid == self.roi_uuid), None)
		if roi is None:
			return
		
		# TODO: From Candidate --> Confirmed
		if self.is_candidate:
			if self.hit_streak >= GMO.min_hit_streak and \
				roi.is_bbox_in_or_touch_roi(bbox_xyxy=self.current_bbox, compute_distance=True) >= GMO.min_entering_distance and \
				self.travelled_distance >= GMO.min_traveled_distance:
				self.moving_state = MovingState.Confirmed
		
		# TODO: From Confirmed --> Counting
		elif self.is_confirmed:
			if roi.is_bbox_in_or_touch_roi(bbox_xyxy=self.current_bbox) <= 0:
				self.moving_state = MovingState.Counting
			
		# TODO: From Counting --> ToBeCounted
		elif self.is_counting:
			if roi.is_center_in_or_touch_roi(bbox_xyxy=self.current_bbox) < 0 or \
				self.time_since_update >= GMO.max_age:
				self.moving_state = MovingState.ToBeCounted
			
		# TODO: From ToBeCounted --> Counted
		# Perform when counting the vehicle
	
		# TODO: From Counted --> Exiting
		elif self.is_counted:
			if roi.is_center_in_or_touch_roi(bbox_xyxy=self.current_bbox, compute_distance=True) <= 0 or \
				self.time_since_update >= GMO.max_age:
				self.moving_state = MovingState.Exiting

	# MARK: Visualize

	def draw(self, drawing, **kwargs):
		"""Draw the object into the ``drawing`` based on the current moving_state.
		
		Args:
			**kwargs: Same as ``general_object.GeneralObject.draw()``
		"""
		if self.is_confirmed:
			GeneralObject.draw(self, drawing=drawing, label=False, **kwargs)
		elif self.is_counting:
			GeneralObject.draw(self, drawing=drawing, label=True, **kwargs)
		elif self.is_counted:
			GeneralObject.draw(self, drawing=drawing, label=True, trajectory=True, color=AppleRGB.values()[self.moi_uuid], **kwargs)
		elif self.is_exiting:
			GeneralObject.draw(self, drawing=drawing, label=True, trajectory=True, color=AppleRGB.values()[self.moi_uuid], **kwargs)
