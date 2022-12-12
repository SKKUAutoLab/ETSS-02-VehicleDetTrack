# ==================================================================== #
# File name: gmo.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/31/2021
# ==================================================================== #
# from __future__ import annotations

from typing import List

from tfe.camera.roi import ROI
from tfe.detector import Detection
from tfe.ops import AppleRGB
from tfe.road_objects.general_object import GeneralObject
from tfe.road_objects.motion_model import MotionModel
from tfe.road_objects.moving_model import MovingModel
from tfe.road_objects.moving_model import MovingState


# MARK: - GMO (General Moving Object)

class GMO(GeneralObject, MotionModel, MovingModel):
	# MARK: Class Property
	
	min_entering_distance: int = 0
	min_traveled_distance: int = 100
	min_hit_streak       : int = 10
	max_age              : int = 1
	
	# MARK: Magic Functions
	
	def __init__(self, **kwargs):
		GeneralObject.__init__(self, **kwargs)
		MotionModel.__init__(self, **kwargs)
		MovingModel.__init__(self, **kwargs)
		
	# MARK: Configure
	
	@classmethod
	def gmo_from_detection(cls, detection: Detection, **kwargs):

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
		# TODO: First update ``GeneralObject``
		self.update_go_from_detection(detection=detection)
		
		# TODO: Second, update motion model
		self.update_motion_state()
		
	def update_moving_state(self, rois: List[ROI], **kwargs):
		roi = next((roi for roi in rois if roi.uuid == self.roi_uuid), None)
		if roi is None:
			return
		
		if self.is_candidate:
			if self.hit_streak >= GMO.min_hit_streak and \
				roi.is_bbox_in_or_touch_roi(bbox_xyxy=self.current_bbox, compute_distance=True) >= GMO.min_entering_distance and \
				self.travelled_distance >= GMO.min_traveled_distance:
				self.moving_state = MovingState.Confirmed
		
		elif self.is_confirmed:
			if roi.is_bbox_in_or_touch_roi(bbox_xyxy=self.current_bbox) <= 0:
				self.moving_state = MovingState.Counting
			
		elif self.is_counting:
			if roi.is_center_in_or_touch_roi(bbox_xyxy=self.current_bbox) < 0 or \
				self.time_since_update >= GMO.max_age:
				self.moving_state = MovingState.ToBeCounted
			
		elif self.is_counted:
			if roi.is_center_in_or_touch_roi(bbox_xyxy=self.current_bbox, compute_distance=True) <= 0 or \
				self.time_since_update >= GMO.max_age:
				self.moving_state = MovingState.Exiting

	# MARK: Visualize

	def draw(self, drawing, **kwargs):
		if self.is_confirmed:
			GeneralObject.draw(self, drawing=drawing, label=False, **kwargs)
		elif self.is_counting:
			GeneralObject.draw(self, drawing=drawing, label=True, **kwargs)
		elif self.is_counted:
			GeneralObject.draw(self, drawing=drawing, label=True, trajectory=True, color=AppleRGB.values()[self.moi_uuid], **kwargs)
		elif self.is_exiting:
			GeneralObject.draw(self, drawing=drawing, label=True, trajectory=True, color=AppleRGB.values()[self.moi_uuid], **kwargs)
