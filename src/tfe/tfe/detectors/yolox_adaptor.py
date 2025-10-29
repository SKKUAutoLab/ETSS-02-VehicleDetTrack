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

from loguru import logger
from torch import Tensor
import torch
from tfe.utils.image import is_channel_first
from tfe.factory.builder import DETECTORS
from tfe.objects.instance import Instance
from tfe.detectors import BaseDetector

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp.build import get_exp_by_name,get_exp_by_file
from yolox.utils import postprocess

__all__ = [
	"YOLOX_Adapter"
]


# MARK: - YOLOX

@DETECTORS.register(name="yolox")
class YOLOX_Adapter(BaseDetector):

	# MARK: Magic Functions

	def __init__(self,
	             name: str = "yolox-x",
	             *args, **kwargs):
		super().__init__(name=name, *args, **kwargs)


	# MARK: Configure

	def init_model(self):
		"""Create and load model from weights."""
		# NOTE: Create model

		# Get image size of detector
		self.exp  = get_exp(None, self.variant)
		self.exp.test_conf   = self.min_confidence
		self.exp.nmsthre     = self.nms_max_overlap
		self.exp.num_classes = self.class_labels.num_classes()

		# Get image size of detector
		if is_channel_first(np.ndarray(self.shape)):
			self.exp.test_size = self.shape[2]
		else:
			self.exp.test_size = self.shape[0]

		self.model = self.exp.get_model()
		self.model.load_state_dict(torch.load(self.weights, map_location=self.device)["model"])
		self.model.to(self.device)
		self.model.eval()

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
			logger.error("Model has not been defined yet!")
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