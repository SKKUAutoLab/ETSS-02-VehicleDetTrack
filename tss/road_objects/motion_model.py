# ==================================================================== #
# File name: motion_model.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/31/2021
#
# Subclassing guide:
# 1. The package (i.e, the .py filename) should be in the template:
#    {motion_model}_{feature_used_to_track}
# ==================================================================== #
import abc

from tss.utils import printe


# MARK: - MotionModel

class MotionModel(metaclass=abc.ABCMeta):
	"""Motion Model.
	
	This class represents the motion model of an individual tracked object. It is used for tracking the moving object.
	
	Attributes:
		hits (int):
			The number of frame has that track appear
		hit_streak (int):
			The number of "consecutive"'" frame has that track appear
		age (int):
			The number of frame while the track is alive, from Candidate -> Deleted
		time_since_update (int):
			The number of 'consecutive' frame that track disappear
		history ():
			Store all the 'predict' position of track in z-bouding box value,
				these position appear while no bounding matches the track
				if any bounding box matches the track, then history = []
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		hits             : int = 0,
		hit_streak       : int = 0,
		age              : int = 0,
		time_since_update: int = 0,
		**kwargs
	):
		self.hits              = hits
		self.hit_streak        = hit_streak
		self.age               = age
		self.time_since_update = time_since_update
		self.history           = []
	
	# MARK: Property
	
	@property
	def matching_features(self):
		"""Return the features used to matched tracked objects with new detections.
		"""
		printe("``matching_features()`` has not been implemented yet")
		raise NotImplementedError

	# MARK: Update
	
	def update_motion_state(self, **kwargs):
		"""Updates the state of the motion model with observed features.
		
		Args:
			Input the specific features used to update the motion model.
		"""
		printe("``update()`` has not been implemented yet")
		raise NotImplementedError
	
	def predict_motion_state(self):
		"""Advances the state of the motion model and returns the predicted estimate.
		"""
		printe("``predict()`` has not been implemented yet")
		raise NotImplementedError

	def current_motion_state(self):
		"""
		Returns the current motion model estimate.
		"""
		printe("``current_estimate()`` has not been implemented yet")
		raise NotImplementedError
