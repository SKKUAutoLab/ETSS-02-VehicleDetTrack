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
import sys
from typing import List
from typing import Optional
from typing import Union
from timeit import default_timer as timer
from collections import Counter

import cv2
import numpy as np

from analyses.analyzer import BaseAnalyzer
from core.factory.builder import ANALYZERS

__all__ = [
	"DriverViolation"
]

from core.objects.moving_model import MovingState


# MARK: - DriverViolation

@ANALYZERS.register(name="driver_violation")
class DriverViolation(BaseAnalyzer):
	"""Analyzer for Driver
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			name          : str             = "driver_violation",
			num_max_people: Optional[int]   = 3,
			num_max_lane  : Optional[int]   = 1,
			ratio_appear  : Optional[float] = 0.1,
			*args, **kwargs
	):
		super().__init__(name=name, *args, **kwargs)
		self.num_max_people               = num_max_people
		self.num_max_lane                 = num_max_lane
		self.ratio_appear                 = ratio_appear
		self.number_people_violation_list = []
		self.number_people_violation_ids  = []
		self.helmet_violation_list        = []
		self.helmet_violation_ids         = []

	# MARK: Property

	# MARK: processing

	def update(self, gmos):
		"""Update the violation

		Args:
			gmos (list):
				Tracking and Matching result
		"""
		# NOTE: check violation of the driver
		self.helmet_violation(gmos)
		self.movement_violation(gmos)
		self.number_people_violation(gmos)

		# return gmos

	def movement_violation(self, gmos):
		"""Find out the movement of motorbike is wrong land

		Args:
			gmos (list):
				Tracking and Matching result
		"""
		for gmo in gmos:
			if gmo.moving_state in [MovingState.Counting, MovingState.ToBeCounted, MovingState.Counted, MovingState.Exiting]:

				# NOTE: check the change lane
				# load the number of lane motorbike going through
				num_lanes    = 0
				counter_lane = Counter(gmo.lanes_id)
				for key, value in counter_lane.items():
					if key is not None:
						num_lanes += 1

				# check the number of people on motorbike is over the maximum
				if num_lanes > self.num_max_lane:
					gmo.is_violated_movement = True
				else:
					gmo.is_violated_movement = False

				# NOTE: check drive in wrong lane
				# check the lane of driver is the same lane as movement or not
				if gmo.moi_uuid is not None:
					if not gmo.is_violated_movement:
						for key, value in counter_lane.items():
							if key is not None and gmo.moi_uuid != key:
								gmo.is_violated_movement = True

	def number_people_violation(self, gmos):
		"""Find out the number of people on motorbike is over the maximum

		Args:
			gmos (list):
				Tracking and Matching result
		"""
		for gmo in gmos:
			if gmo.moving_state in [MovingState.Counting, MovingState.ToBeCounted, MovingState.Counted, MovingState.Exiting]:

				# load the number of people on motorbike
				list_num_person = []
				for instances_in_bbox in gmo.instances_in_bboxes:
					# list_num_person.append(len(instances_in_bbox))
					list_num_person.append(len(instances_in_bbox))
					# print(len(instances_in_bbox))

				# get the number of people on motorbike through out the trajectory
				max_num_appear_ = 0
				for number_of_people, num_appear_in_bounding_box in Counter(list_num_person).items():
					if num_appear_in_bounding_box > max_num_appear_:
						max_num_appear_ = num_appear_in_bounding_box
						gmo.num_people  = number_of_people

				# check the number of people on motorbike is over the maximum
				if gmo.num_people > self.num_max_people + 1:
					gmo.is_violated_num_people = True
				else:
					gmo.is_violated_num_people = False

				# DEBUG:
				# print("*******************************")
				# print(f"{len(gmo.bboxes_id)}")
				# print(f"{len(gmo.instances_in_bboxes)}")
				# print(Counter(list_num_person))
				# print(type(Counter(list_num_person)))
				# for key, value in Counter(list_num_person).items():
				# 	print(f"{key} :: {value}")
				# print("*******************************")
				# print(list_num_person)
				# print(dir(gmo))
				# gmo.is_violated_num_people = False
				# print(gmo.lalala)
				# print(gmo.num_people)
				# print(gmo.helmets)
				# print(gmo.ratio_appear)
				# print(gmo.is_violated_num_people)
				# sys.exit()

				# NOTE: save violated gmo
				if gmo.is_violated_num_people:
					if gmo.id not in self.number_people_violation_ids:
						self.number_people_violation_ids.append(gmo.id)
				# 		self.number_people_violation_list.append(gmo)
					# print(gmo.id)
					# print(Counter(list_num_person))

	def helmet_violation(self, gmos):
		"""Find out non-helmet wearing

		Args:
			gmos (list):
				Tracking and Matching result
		"""
		# list of class is used in AIC23
		classes_aic23 = ['motorbike',
						 'DHelmet', 'DNoHelmet',
						 'P1Helmet', 'P1NoHelmet',
						 'P2Helmet', 'P2NoHelmet']

		for gmo in gmos:
			if gmo.moving_state in [MovingState.Counting, MovingState.ToBeCounted, MovingState.Counted, MovingState.Exiting]:

				helmet_check        = np.zeros(len(classes_aic23))
				len_gmo_ratio_check = int(len(gmo.instances_in_bboxes) * self.ratio_appear)

				# load the helmet on motorbike
				for instances_in_bbox in gmo.instances_in_bboxes:
					for instance_in_bbox in instances_in_bbox:
						helmet_check[instance_in_bbox.class_id] += 1

				if helmet_check[2] > len_gmo_ratio_check or \
						helmet_check[4] > len_gmo_ratio_check or \
						helmet_check[6] > len_gmo_ratio_check:
					gmo.is_violated_helmet = True
				else:
					gmo.is_violated_helmet = False

				# save violated gmo
				if gmo.is_violated_helmet:
					if gmo.id not in self.helmet_violation_ids:
						self.helmet_violation_ids.append(gmo.id)
				# 		self.helmet_violation_list.append(gmo)
						# print(gmo.id)
						# print(helmet_check)
