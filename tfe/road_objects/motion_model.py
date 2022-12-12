# ==================================================================== #
# File name: motion_model.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/31/2021
# ==================================================================== #
import abc

from tfe.utils import printe


# MARK: - MotionModel

class MotionModel(metaclass=abc.ABCMeta):

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
		printe("``matching_features()`` has not been implemented yet")
		raise NotImplementedError

	# MARK: Update
	
	def update_motion_state(self, **kwargs):
		printe("``update()`` has not been implemented yet")
		raise NotImplementedError
	
	def predict_motion_state(self):
		printe("``predict()`` has not been implemented yet")
		raise NotImplementedError

	def current_motion_state(self):
		printe("``current_estimate()`` has not been implemented yet")
		raise NotImplementedError
