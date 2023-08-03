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

import cv2
import numpy as np
from munch import Munch
from core.factory.builder import MATCHERS

from matcher import BaseMatcher
from matcher.moi import MOI
from matcher.roi import ROI

__all__ = [
	"MatcherRnT"
]


# MARK: - YOLOv8

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
			**kwargs
	):
		super().__init__(name=name, **kwargs)
		self.moi_cfg  = moi
		self.roi_cfg  = roi
		self.rois     = None
		self.mois     = None

		# NOTE: Load components
		self.load_mois(self.moi_cfg["file"])
		self.load_rois(self.roi_cfg["file"])

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

	# MARK: Processing


