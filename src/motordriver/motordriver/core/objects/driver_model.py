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
import enum
from typing import Optional

from core.utils.rich import console, error_console

import numpy as np

__all__ = [
	"DriverModel"
]


class ViolationState(enum.Enum):
	"""An enum that identify the counting state of an object when moving through the camera."""
	Candidate   = 1  # Preliminary state.
	Confirmed   = 2  # Confirmed the Detection is a road_objects eligible for counting.
	Counting    = 3  # Object is in the counting zone/counting state.


class DriverModel(object):
	"""Moving Model for matching (or Flow Estimation)
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			num_people : Optional[int]        = 0,
			helmets    : Optional[np.ndarray] = None,
			**kwargs
	):
		self.num_people = num_people
		self.helmets    = helmets if (helmets is not None) else np.zeros(self.num_people)

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

	# MARK: Update

	def update(self):
		pass
