from __future__ import annotations

import os
import platform
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess

from tfe.detectors import BaseDetector
from tfe.factory.builder import DETECTORS
from tfe.objects.instance import Instance
from tfe.utils.image import is_channel_first

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
		self.exp             = get_exp(exp_file=self.variant)
		self.exp.test_conf   = self.min_confidence
		self.exp.nmsthre     = self.nms_max_overlap
		self.exp.num_classes = self.class_labels.num_classes()

		# Get image size of detector
		if is_channel_first(np.ndarray(self.shape)):
			self.exp.test_size = [self.shape[2], self.shape[1]]
		else:
			self.exp.test_size = [self.shape[1], self.shape[0]]

		self.rgb_means = (0.485, 0.456, 0.406)
		self.std = (0.229, 0.224, 0.225)

		self.model = self.exp.get_model()
		checkpoint = torch.load(self.weights, map_location=self.device, weights_only=False)
		self.model.load_state_dict(checkpoint["model"])
		self.model.to(self.device)
		self.model.eval()
		if self.half:
			self.model = fuse_model(self.model).half()  # Conv+BN fusion with FP16, as YOLOX's demo --fp16 --fuse
		self.ratio = 1.0

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

	def preprocess(self, images: list) -> Tensor:
		"""Letterbox the frames as ``yolox.data.data_augment.preproc`` does, but in uint8,
		and cast them on the device: a quarter of the bytes to upload.

		Args:
			images (list):
				Frames of shape [H, W, C], BGR uint8.

		Returns:
			input (Tensor):
				[B, C, H, W] on ``self.device``: float16 when ``self.half``, else float32 equal to ``preproc``'s output.
		"""
		height, width = self.exp.test_size  # the (rows, cols) order preproc uses
		batch = np.full((len(images), height, width, 3), 114, dtype=np.uint8)
		for i, image in enumerate(images):
			self.ratio = min(height / image.shape[0], width / image.shape[1])
			resized    = cv2.resize(image, (int(image.shape[1] * self.ratio), int(image.shape[0] * self.ratio)), interpolation=cv2.INTER_LINEAR)
			batch[i, :resized.shape[0], :resized.shape[1]] = resized
		input_imgs = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).contiguous()
		return input_imgs.half() if self.half else input_imgs.float()

	def forward(self, input_imgs: Tensor) -> list:
		"""One forward pass and one NMS call for the whole batch.

		Args:
			input_imgs (Tensor):
				Input images of shape [B, C, H, W].

		Returns:
			pred (list):
				One [N, 7] ndarray per image (x1, y1, x2, y2, obj_conf, class_conf, class);
				[0, 7] when nothing is detected (yolox's postprocess gives None there).
		"""
		outputs = postprocess(self.model(input_imgs).float(), self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre)  # NMS in float32, also under FP16
		return [np.empty((0, 7), dtype=np.float32) if output is None else output.cpu().numpy() for output in outputs]

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
			indexes (np.ndarray): order of images
				Image indexes.
			images (np.ndarray): raw images
				Images of shape [B, H, W, C].
			input_imgs (Tensor): pre_processed images
				Input image of shape [B, C, H, W].
			pred (Tensor): results from model
				Prediction.

		Returns:
			instances (list):
				List of `Instances` objects.
		"""
		# NOTE: Create Detection objects
		instances = []

		for idx, (frame_index, result) in enumerate(zip(indexes, pred)):
			inst  = []
			xyxys = result[: , 0 : 4] / self.ratio
			confs = result[: , 4] * result[: , 5]
			clses = result[: , 6]

			mask  = confs >= self.min_confidence
			xyxys = xyxys[mask]
			confs = confs[mask]
			clses = clses[mask]

			for bbox_xyxy, conf, cls in zip(xyxys, confs, clses):
				confident   = float(conf)
				class_id    = int(cls)
				class_label = self.class_labels.get_class_label(
					key="train_id", value=class_id
				)
				inst.append(
					Instance(
						frame_index = frame_index,
						bbox        = bbox_xyxy,
						confidence  = confident,
						class_label = class_label,
						label       = class_label,
						class_id    = class_id
					)
				)

			instances.append(inst)
		return instances
