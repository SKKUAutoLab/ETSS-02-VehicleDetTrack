#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Detector classes.
"""

from __future__ import annotations

from typing import Dict

from munch import Munch
from loguru import logger

from .basedetector import BaseDetector
from .yolov8_adaptor import YOLOv8_Adapter
from .yolov11_adaptor import YOLOv11_Adapter

def get_detector(hparams: Dict, **kwargs) -> BaseDetector:
	"""Get the detector model based on the given hyperparameters.

	Args:
		hparams (dict):
			The model's hyperparameters.

	Returns:
		detector (Detector):
			The detector model.
	"""
	hparams = hparams if isinstance(hparams, Munch) else Munch(hparams)
	name    = hparams.name

	if name == "yolov8":  # NOTE: the version for AI City Challenge 2021.
		return YOLOv8_Adapter(**hparams, **kwargs)
	elif name == "yolov11":
		return YOLOv11_Adapter(**hparams, **kwargs)
	else :
		logger.error(f"Unsupported detector model: {name}.")