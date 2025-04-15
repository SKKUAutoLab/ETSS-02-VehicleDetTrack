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

import enum
import abc
from typing import Optional

import numpy as np

__all__ = [
	"MotorbikeDriverModel"
]


class ViolationState(enum.Enum):
	"""An enum that identify the counting state of an object when moving through the camera."""
	Candidate   = 1  # Preliminary state.
	Confirmed   = 2  # Confirmed the Detection is a road_objects eligible for counting.
	Counting    = 3  # Object is in the counting zone/counting state.


class MotorbikeDriverModel(object):
	"""Motorbike Model for check the motorbike driver status
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			num_people            : Optional[int]        = 0,
			helmets               : Optional[np.ndarray] = None,
			ratio_appear          : Optional[float]      = 0.1,
			is_violated_num_people: Optional[bool]       = False,
			is_violated_helmet    : Optional[bool]       = False,
			is_violated_movement  : Optional[bool]       = False,
			**kwargs
	):
		"""

		Args:
			num_people (int, optional):
				The number of people on the motorbike. Defaults to 0.
			helmets:
			ratio_appear:
			is_violated_num_people:
			is_violated_helmet:
			is_violated_movement:
			**kwargs:
		"""
		# super().__init__(**kwargs)
		self.num_people   = num_people
		self.ratio_appear = ratio_appear
		self.helmets      = helmets

		# NOTE: check the violated state
		self.is_violated_num_people = is_violated_num_people
		self.is_violated_helmet     = is_violated_helmet
		self.is_violated_movement   = is_violated_movement

	# MARK: Property

	@property
	def num_people(self):
		return self._num_people

	@num_people.setter
	def num_people(self, num_people: int):
		self._num_people = num_people

	@property
	def helmets(self):
		return self._helmets

	@helmets.setter
	def helmets(self, helmets: list):
		self._helmets = helmets

	@property
	def is_violated_num_people(self):
		return self._is_violated_num_people

	@is_violated_num_people.setter
	def is_violated_num_people(self, value):
		self._is_violated_num_people = value

	@property
	def is_violated_helmet(self):
		return self._is_violated_helmet

	@is_violated_helmet.setter
	def is_violated_helmet(self, is_violated_helmet: bool):
		self._is_violated_helmet = is_violated_helmet

	@property
	def is_violated_movement(self):
		return self._is_violated_movement

	@is_violated_movement.setter
	def is_violated_movement(self, is_violated_movement: bool):
		self._is_violated_movement = is_violated_movement

	# MARK: Update

	# def update(self):
	# 	pass
