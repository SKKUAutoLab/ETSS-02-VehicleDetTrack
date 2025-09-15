from __future__ import annotations

import sys
import warnings
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import platform
from torch import Tensor
import torch
from thermal_pedestrian.core.utils.image import is_channel_first
from thermal_pedestrian.core.factory.builder import DETECTORS
from thermal_pedestrian.core.objects.instance import Instance
from thermal_pedestrian.detectors import BaseDetector

from ultralytics.models.yolo import detect

__all__ = [
	"YOLOv8_Adapter"
]


# MARK: - YOLOv8

@DETECTORS.register(name="yolov8")
class YOLOv8_Adapter(BaseDetector):

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
			'imgsz'   : self.img_size,
			'conf'    : self.min_confidence,
			'iou'     : self.nms_max_overlap,
			'show'    : False,
			'verbose' : False,
			'save'    : False,
			'device'  : self.device,
			'max_det' : 300
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
		input_imgs = self.preprocess(images=images)
		# NOTE: Forward
		preds  = self.forward(input_imgs)
		# NOTE: Postprocess
		instances = self.postprocess(
			indexes=indexes, images=images, input_imgs=input_imgs, pred=preds
		)
		# NOTE: Suppression
		# FIXME: no co van de nghiem trong, chi chay 7cls chu khong phai la 9cls
		# allows_ids trong class_label co van de
		# instances = self.suppress_wrong_labels(instances=instances)

		return instances

	def preprocess(self, images: np.ndarray):
		"""Preprocess the input images to model's input image.

		Args:
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			input (Tensor):
				Models' input.
		"""
		input_imgs = images
		# if self.shape:
		# 	input = padded_resize(input, self.shape, stride=self.stride)
		# 	self.resize_original = True
		# #input = [F.to_tensor(i) for i in input]
		# #input = torch.stack(input)
		# input = to_tensor(input, normalize=True)
		# input = input.to(self.device)
		return input_imgs

	def forward(self, input_imgs: Tensor):
		"""Forward pass.

		Args:
			input_imgs (Tensor):
				Input image of shape [B, C, H, W].

		Returns:
			pred (Tensor):
				Predictions.
		"""
		pred = self.model(source=input_imgs)
		return pred

	def postprocess(
			self,
			indexes   : np.ndarray,
			images    : np.ndarray,
			input_imgs: Tensor,
			pred      : Tensor,
			*args, **kwargs
	) -> list:
		"""Postprocess the prediction.

		Args:
			indexes (np.ndarray):
				Image indexes.
			images (np.ndarray):
				Images of shape [B, H, W, C].
			input_imgs (Tensor):
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

		for idx, (frame_index, result) in enumerate(zip(indexes, pred)):
			inst = []
			xyxyns = result.boxes.xywhn.cpu().numpy()
			confs = result.boxes.conf.cpu().numpy()
			clses = result.boxes.cls.cpu().numpy()

			for bbox_xyxyn, conf, cls in zip(xyxyns, confs, clses):
				confident   = float(conf)
				class_id    = int(cls)
				class_label = self.class_labels.get_class_label(
					key="train_id", value=class_id
				)
				inst.append(
					Instance(
						frame_index = frame_index,
						bbox        = bbox_xyxyn,
						confidence  = confident,
						class_label = class_label,
						label       = class_label,
						class_id    = class_id
					)
				)

			instances.append(inst)
		return instances