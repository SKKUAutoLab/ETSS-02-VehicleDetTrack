#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YOLOv5 object_detectors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import platform
from torch import Tensor

from core.utils.image import is_channel_first
from core.factory.builder import DETECTORS
from core.objects.instance import Instance
from detectors.detector import BaseDetector

# NOTE: add PATH of YOLOv8 source to here
FILE = Path(__file__).resolve()
ROOT = os.path.join(FILE.parents[0].parents[0])  # YOLOv8 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.models.yolo import detect

__all__ = [
	"YOLOv8"
]


# MARK: - YOLOv8

@DETECTORS.register(name="yolov8")
class YOLOv8(BaseDetector):
	"""YOLOv8 object detector."""

	# MARK: Magic Functions

	def __init__(self,
				 name: str = "yolov8",
				 *args, **kwargs):
		super().__init__(name=name, *args, **kwargs)

	# MARK: Configure

	def init_model(self):
		"""Create and load model from weights."""
		# NOTE: Create model
		# path = self.weights
		# if not is_torch_saved_file(path):
		# 	path, _ = os.path.splitext(path)
		# 	path    = os.path.join(models_zoo_dir, f"{path}.pt")
		# assert is_torch_saved_file(path), f"Not a weights file: {path}"

		# Get image size of detector
		if is_channel_first(np.ndarray(self.shape)):
			self.img_size = self.shape[2]
		else:
			self.img_size = self.shape[0]

		# NOTE: load model
		self.model = detect.DetectionPredictor(overrides={
			'imgsz'  : self.img_size,
			'conf'   : self.min_confidence,
			'iou'    : self.nms_max_overlap,
			'show'   : False,
			'verbose': False,
			'save'   : False,
			'device' : self.device
		})
		_ = self.model(source=np.zeros([self.img_size, self.img_size, 3]), model=self.weights)

	# MARK: Detection

	def detect(self, indexes: np.ndarray, images: np.ndarray) -> list:
		"""Detect objects in the images.

		Args:
			indexes (np.ndarray):
				Image indexes.
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			instances (list):
				List of `Instance` objects.
		"""
		# NOTE: Safety check
		if self.model is None:
			print("Model has not been defined yet!")
			raise NotImplementedError

		# NOTE: Preprocess
		input = self.preprocess(images=images)
		# NOTE: Forward
		pred  = self.forward(input)
		# NOTE: Postprocess
		instances = self.postprocess(
			indexes=indexes, images=images, input=input, pred=pred
		)
		# NOTE: Suppression
		instances = self.suppress_wrong_labels(instances=instances)

		return instances

	def preprocess(self, images: np.ndarray) -> Tensor:
		"""Preprocess the input images to model's input image.

		Args:
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			input (Tensor):
				Models' input.
		"""
		input = images
		# if self.shape:
		# 	input = padded_resize(input, self.shape, stride=self.stride)
		# 	self.resize_original = True
		# #input = [F.to_tensor(i) for i in input]
		# #input = torch.stack(input)
		# input = to_tensor(input, normalize=True)
		# input = input.to(self.device)
		return input

	def forward(self, input: Tensor) -> Tensor:
		"""Forward pass.

		Args:
			input (Tensor):
				Input image of shape [B, C, H, W].

		Returns:
			pred (Tensor):
				Predictions.
		"""
		pred = self.model(source=input)
		return pred

	def postprocess(
			self,
			indexes: np.ndarray,
			images : np.ndarray,
			input  : Tensor,
			pred   : Tensor,
			*args, **kwargs
	) -> list:
		"""Postprocess the prediction.

		Args:
			indexes (np.ndarray):
				Image indexes.
			images (np.ndarray):
				Images of shape [B, H, W, C].
			input (Tensor):
				Input image of shape [B, C, H, W].
			pred (Tensor):
				Prediction.

		Returns:
			instances (list):
				List of `Instances` objects.
		"""
		# NOTE: Create Detection objects
		instances = []
		# DEBUG:
		# print("******")
		# for result in pred:
		# 	# detection
		# 	result.boxes.xyxy  # box with xyxy format, (N, 4)
		# 	result.boxes.xywh  # box with xywh format, (N, 4)
		# 	result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
		# 	result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
		# 	result.boxes.conf  # confidence score, (N, 1)
		# 	result.boxes.cls  # cls, (N, 1)
		# print("******")

		for idx, result in enumerate(pred):
			inst = []
			xyxys = result.boxes.xyxy.cpu().numpy()
			confs = result.boxes.conf.cpu().numpy()
			clses = result.boxes.cls.cpu().numpy()

			for bbox_xyxy, conf, cls in zip(xyxys, confs, clses):
				confident   = float(conf)
				class_id    = int(cls)
				class_label = self.class_labels.get_class_label(
					key="train_id", value=class_id
				)
				inst.append(
					Instance(
						frame_index = indexes[0] + idx,
						bbox        = bbox_xyxy,
						confidence  = confident,
						class_label = class_label,
						label       = class_label
					)
				)

			instances.append(inst)
		return instances


