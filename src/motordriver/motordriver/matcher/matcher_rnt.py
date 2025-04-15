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
from munch import Munch
from core.factory.builder import MATCHERS
from core.objects.moving_model import MovingState

from matcher import BaseMatcher
from matcher.moi import MOI
from matcher.loi import LOI
from matcher.roi import ROI

__all__ = [
	"MatcherRnT"
]


# MARK: - Matcher_RNT

@MATCHERS.register(name="matcher_rnt")
class MatcherRnT(BaseMatcher):
	"""Matcher (Match for flow)
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			name: str = "matcher_rnt",
			moi : Optional[dict] = None,
			roi : Optional[dict] = None,
			loi : Optional[dict] = None,
			**kwargs
	):
		super().__init__(name=name, **kwargs)
		self.moi_cfg            = moi
		self.roi_cfg            = roi
		self.loi_cfg            = loi
		self.rois               = None
		self.mois               = None
		self.lois               = None
		self.in_roi_gmos        = None
		self.countable_gmos     = None
		self.to_be_counted_gmos = None
		self.is_counted_gmos    = None

		# NOTE: Load components
		self.load_mois(self.moi_cfg["file"])
		self.load_rois(self.roi_cfg["file"])
		self.load_lois(self.loi_cfg["file"])

	# MARK: Property

	# MARK: Configure

	def load_rois(self, rois: dict):
		"""Load the list of region of interest

		Args:
			rois (dict):
				List of path

		"""
		self.rois = []
		for roi_path in rois:
			self.rois = self.rois + ROI.load_rois_from_file(
									dataset = self.roi_cfg["dataset"],
									file    = roi_path
								)

	def load_mois(self, mois: dict):
		"""Load the list of movement of interest

		Args:
			mois (dict):
				List of path

		"""
		self.mois = []
		for moi_path in mois:
			self.mois = self.mois + MOI.load_mois_from_file(
					dataset = self.roi_cfg["dataset"],
					file    = moi_path
				)

	def load_lois(self, lois: dict):
		"""Load the list of movement of interest

		Args:
			lois (dict):
				List of path

		"""
		self.lois = []
		for loi_path in lois:
			self.lois = self.lois + LOI.load_lois_from_file(
					dataset = self.roi_cfg["dataset"],
					file    = loi_path
			)

	# MARK: Processing

	def update(self, gmos):
		"""Update the state of the moving objects and perform counting.

		Args:
			gmos (list):
				The moving objects to be updated.
		"""
		self.update_moving_state(gmos)
		self.associate_gmos_with_mois(gmos)
		self.counting(gmos)

	def update_lane(self, gmos):
		"""
		Update the lane of the moving objects.

		Args:
			gmos:
				The moving objects to be updated.
		"""
		self.associate_gmos_with_lois(gmos)

	def update_moving_state(self, gmos):
		"""Update the moving state of the moving objects.

		Args:
			gmos (list):
				The moving objects to be updated
		Returns:

		"""
		for gmo in gmos:
			gmo.update_moving_state(rois=self.rois)
			gmo.timestamps.append(timer())

	def associate_gmos_with_mois(self, gmos):
		"""Associate the moving objects with the movements of interest.

		Args:
			gmos (list):
				The moving objects to be associated.
		"""
		# NOTE: Associate gmos with MOIs
		self.in_roi_gmos = [o for o in gmos if o.is_confirmed or o.is_counting or o.is_to_be_counted]
		MOI.associate_moving_objects_to_mois(
			gmos=self.in_roi_gmos, mois=self.mois, shape_type="polygon")
		self.to_be_counted_gmos = [o for o in self.in_roi_gmos if o.is_to_be_counted and o.is_countable is False]
		MOI.associate_moving_objects_to_mois(
			gmos=self.to_be_counted_gmos, mois=self.mois, shape_type="linestrip")

	def associate_gmos_with_lois(self, gmos):
		"""Associate the moving objects with the lines of interest.

		Args:
			gmos (list):
				The moving objects to be associated.
		"""
		self.is_counted_gmos = [o for o in gmos]
		for gmo in self.is_counted_gmos:
			gmo.lanes_id = [None for _ in range(len(gmo.bboxes))]
		LOI.associate_moving_objects_to_lois(
			gmos=self.is_counted_gmos, lois=self.lois, shape_type="polygon")

	def counting(self, gmos):
		self.countable_gmos = [o for o in self.in_roi_gmos if (o.is_countable and o.is_to_be_counted)]
		for gmo in self.countable_gmos:
			gmo.moving_state = MovingState.Counted
