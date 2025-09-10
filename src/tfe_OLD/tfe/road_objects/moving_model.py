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
from typing import Optional

from tfe.utils import printe


# MARK: - MovingState

class MovingState(enum.Enum):
	"""An enum that identify the counting state of an object when moving through the camera."""
	Candidate   = 1  # Preliminary state.
	Confirmed   = 2  # Confirmed the Detection is a road_objects eligible for counting.
	Counting    = 3  # Object is in the counting zone/counting state.
	ToBeCounted = 4  # Mark object to be counted somewhere in this loop iteration.
	Counted     = 5  # Mark object has been counted.
	Exiting     = 6  # Mark object for exiting the ROI or image frame.  Let's it die by itself.
	
	
# MARK: - CountingModel

# noinspection PyMissingOrEmptyDocstring
class MovingModel(object):
	"""Moving Model.
	"""
	
	# MARK: Magic Functions

	def __init__(
		self,
		moving_state: MovingState   = MovingState.Candidate,
		roi_uuid    : Optional[int] = None,
		moi_uuid    : Optional[int] = None,
		**kwargs
	):
		self.moving_state = moving_state
		self.roi_uuid     = roi_uuid
		self.moi_uuid     = moi_uuid
	
	# MARK: Property
	
	@property
	def moving_state(self) -> MovingState:
		return self._moving_state
	
	@moving_state.setter
	def moving_state(self, counting_state: MovingState):
		allowed_states = [MovingState.Candidate,   MovingState.Confirmed, MovingState.Counting,
		                  MovingState.ToBeCounted, MovingState.Counted,   MovingState.Exiting]
		if counting_state not in allowed_states:
			printe(f"State should be one of: {allowed_states}. But given {counting_state}.")
			raise ValueError
		self._moving_state = counting_state
	
	@property
	def is_candidate(self) -> bool:
		return self.moving_state == MovingState.Candidate
	
	@property
	def is_confirmed(self) -> bool:
		return self.moving_state == MovingState.Confirmed
	
	@property
	def is_counting(self) -> bool:
		return self.moving_state == MovingState.Counting
	
	@property
	def is_countable(self) -> bool:
		return True if (self.moi_uuid is not None) else False
	
	@property
	def is_to_be_counted(self) -> bool:
		return self.moving_state == MovingState.ToBeCounted
	
	@property
	def is_counted(self) -> bool:
		return self.moving_state == MovingState.Counted
	
	@property
	def is_exiting(self) -> bool:
		return self.moving_state == MovingState.Exiting
	
	# MARK: Update
	
	def update_moving_state(self, **kwargs):
		"""Update the current state of the road_objects.
		"""
		printe("``update_moving_state()`` has not been implemented yet.")
		raise NotImplementedError
