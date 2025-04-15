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
# ==================================================================== #
import abc

__all__ = [
	"MotionModel"
]


# MARK: - MotionModel


class MotionModel(metaclass=abc.ABCMeta):
	"""Object class for tracking
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
		print("``matching_features()`` has not been implemented yet")
		raise NotImplementedError

	# MARK: Update
	
	def update_motion_state(self, **kwargs):
		print("``update()`` has not been implemented yet")
		raise NotImplementedError
	
	def predict_motion_state(self):
		print("``predict()`` has not been implemented yet")
		raise NotImplementedError

	def current_motion_state(self):
		print("``current_estimate()`` has not been implemented yet")
		raise NotImplementedError
