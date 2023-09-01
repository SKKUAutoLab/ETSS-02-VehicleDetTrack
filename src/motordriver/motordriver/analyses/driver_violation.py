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
# from __future__ import annotations

import os
from typing import List
from typing import Optional
from typing import Union
from timeit import default_timer as timer

import cv2
import numpy as np

from analyses.analyzer import BaseAnalyzer
from core.factory.builder import ANALYZERS

__all__ = [
	"DriverViolation"
]


# MARK: - DriverViolation

@ANALYZERS.register(name="driver_violation")
class DriverViolation(BaseAnalyzer):
	"""Analyzer for Driver
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			name: str = "driver_violation",
			**kwargs
	):
		super().__init__(name=name, **kwargs)

	# MARK: Property

	# MARK: processing

	def update(self, gmos, batch_identifications):
		"""Update the violation

		Args:
			gmos (list):
			batch_identifications (list):

		"""
		for gmo in gmos:
			for identification_instance in batch_identifications:
				if gmo.bboxes_id[-1][0] == identification_instance.id[0] and gmo.bboxes_id[-1][1] == \
						identification_instance.id[1]:
					print(identification_instance.id)

