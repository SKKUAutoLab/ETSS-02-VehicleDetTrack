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
	"RouteViolation"
]

from core.objects.moving_model import MovingState

# MARK: - RouteViolation

@ANALYZERS.register(name="route_violation")
class RouteViolation(BaseAnalyzer):
	"""Analyzer for Wrong Route
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			name: str = "route_violation",
			*args, **kwargs
	):
		super().__init__(name=name, *args, **kwargs)
		pass

	# MARK: Property

	# MARK: processing

	def update(self, gmos, batch_identifications):
		"""Update the violation

		Args:
			gmos (list):
				Tracking and Matching result
			batch_identifications (list):
				Identifier result
		"""

		# NOTE: sync tracking with identification
		# id: [frame_index, bounding_box index, instance_index]
		for gmo in gmos:
			for identification_instance in batch_identifications:
				if (gmo.bboxes_id[-1][0] == identification_instance.id[0] and
						gmo.bboxes_id[-1][1] == identification_instance.id[1]):

					# add person on the motorbike
					gmo.num_people += 1

					# DEBUG:
					# print(identification_instance.id)
					# if gmo.moving_state == MovingState.ToBeCounted:
					# 	print(gmo.id)
					# 	print(gmo.moving_state)
					# 	for bbox_id in gmo.bboxes_id:
					# 		print(bbox_id)
					# 	for identification_instance_temp in batch_identifications:
					# 		if (gmo.bboxes_id[-1][0] == identification_instance_temp.id[0] and
					# 				gmo.bboxes_id[-1][1] == identification_instance_temp.id[1]):
					# 			print(identification_instance_temp.id)
					#
					# 	sys.exit()

					if gmo.moving_state == MovingState.Counted:

						# DEBUG:
						print(gmo.id)
						# print(gmo.moving_state)

						# for bbox_id in gmo.bboxes_id:
						# 	print(bbox_id)

						for identification_instance_temp in batch_identifications:
							if (gmo.bboxes_id[-1][0] == identification_instance_temp.id[0] and
									gmo.bboxes_id[-1][1] == identification_instance_temp.id[1]):
								print(identification_instance_temp.id)
								print(gmo.bboxes_id)

				# sys.exit()
