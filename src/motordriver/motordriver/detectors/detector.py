#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all detector models. It defines an unify template to
guarantee the input and output of all object_detectors are the same.
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import functional as F

from core.data.class_label import ClassLabels
from core.type.type import Dim3
from core.utils.device import select_device
from core.utils.geometric_transformation import padded_resize

__all__ = [
	"BaseDetector"
]


# MARK: - BaseDetector
from core.utils.image import to_tensor


# noinspection PyShadowingBuiltins
class BaseDetector(metaclass=abc.ABCMeta):
	"""Base Detector.

	Attributes:
		name (str):
			Name of the detector model.
		model (nn.Module):
			Detector model.
		model_cfg (dict, optional):
			Detector model's config.
		class_labels (ClassLabels, optional):
			List of all labels' dicts.
		allowed_ids (np.ndarray, optional):
			Array of all class_labels' ids. Default: `None`.
		weights (str, optional):
			Path to the pretrained weights. Default: `None`.
		shape (Dim3, optional):
			Input size as [H, W, C]. This is also used to resize the image.
			Default: `None`.
		min_confidence (float, optional):
			Detection confidence threshold. Remove all detections that have a
			confidence lower than this value. If `None`, don't perform
			suppression. Default: `0.5`.
		nms_max_overlap (float, optional):
			Maximum detection overlap (non-maxima suppression threshold).
			If `None`, don't perform suppression. Default: `0.4`.
		device (str, optional):
			Cuda device, i.e. 0 or 0,1,2,3 or cpu. Default: `None` means CPU.
		resize_original (bool):
			Should resize the predictions back to original image resolution?
			Default: `False`.
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			name           : str,
			model_cfg      : Optional[dict],
			class_labels   : ClassLabels,
			weights        : Optional[str]   = None,
			shape          : Optional[Dim3]  = None,
			min_confidence : Optional[float] = 0.5,
			nms_max_overlap: Optional[float] = 0.4,
			device         : Optional[str]   = None,
			variant        : Optional[str] = None,
			*args, **kwargs
	):
		super().__init__()
		self.name            = name
		self.model           = None
		self.model_cfg       = model_cfg
		self.variant         = variant if variant is not None else name
		self.class_labels    = class_labels
		self.allowed_ids     = self.class_labels.ids(key="train_id")
		self.weights         = weights
		self.shape           = shape
		self.min_confidence  = min_confidence
		self.nms_max_overlap = nms_max_overlap
		self.device          = select_device(device=device)
		self.resize_original = False
		self.stride          = 32

		# NOTE: Load model
		self.init_model()

	# MARK: Configure

	@abc.abstractmethod
	def init_model(self):
		"""Create and load model from weights."""
		pass

	# MARK: Detection

	@abc.abstractmethod
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
		pass

	@abc.abstractmethod
	def forward(self, input: Tensor) -> Tensor:
		"""Forward pass.

		Args:
			input (Tensor):
				Input image of shape [B, C, H, W].

		Returns:
			pred (Tensor):
				Predictions.
		"""
		pass

	@abc.abstractmethod
	def preprocess(self, images: np.ndarray) -> Tensor:
		"""Preprocess the input images to model's input image.

		Args:
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			input (Tensor):
				Models' input.
		"""
		pass

	@abc.abstractmethod
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
				List of `Instance` objects.
		"""
		pass

	def suppress_wrong_labels(self, instances: list) -> list:
		"""Suppress all instances with wrong labels.

		Args:
			instances (list):
				List of `Instance` objects of shape [B, ...], where B is the
				number of batch.

		Returns:
			valid_instances (list):
				List of valid `Detection` objects of shape [B, ...], where B
				is the number of batch.
		"""
		valid_instances = []
		for inst in instances:
			valid_inst = [i for i in inst if self.is_correct_label(i.class_label)]
			valid_instances.append(valid_inst)
		return valid_instances

	def suppress_low_confident(self, instances: list) -> list:
		"""Suppress instances with low confidence scores.

		Args
			detections (list):
				List of valid `Instance` objects of shape [B, ...], where B is
				the number of batch.

		Returns:
			valid_instances (list):
				List of high-confident `Instance` objects of shape [B, ...],
				where B is the number of batch.
		"""
		if self.min_confidence is None:
			return instances

		valid_instances = []
		for inst in instances:
			valid_inst = [
				i for i in inst if (i.confidence >= self.min_confidence)
			]
			valid_instances.append(valid_inst)
		return valid_instances

	def suppress_non_max(self, instances: list) -> list:
		"""Suppress overlapping detections (high-level). Original code from:
		.. [1] http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

		Args:
			instances (list):
				List of high-confident `Instance` objects of shape [B, ...],
				where B is the number of batch.

		Returns:
			valid_instances (list):
				List of non-overlapped `Instance` objects of shape [B, ...],
				where B is the number of batch.
		"""
		if self.nms_max_overlap is None:
			return instances

		valid_instances = []
		for inst in instances:
			# NOTE: Extract detection bounding boxes and scores
			boxes  = np.array([i.bbox       for i in inst])
			scores = np.array([i.confidence for i in inst])

			# NOTE: Extract road_objects indices that survive
			# non-max-suppression
			indices = []
			if len(boxes) > 0:
				boxes = boxes.astype(np.float)

				# Top-left to Bottom-right
				x1 = boxes[:, 0]
				y1 = boxes[:, 1]
				x2 = boxes[:, 2] + boxes[:, 0]
				y2 = boxes[:, 3] + boxes[:, 1]

				# Area
				area = (boxes[:, 2] + 1) * (boxes[:, 3] + 1)
				if scores is not None:
					idxs = np.argsort(scores)
				else:
					idxs = np.argsort(y2)

				# Suppression via iterating boxes
				while len(idxs) > 0:
					last = len(idxs) - 1
					i    = idxs[last]
					indices.append(i)

					xx1     = np.maximum(x1[i], x1[idxs[:last]])
					yy1     = np.maximum(y1[i], y1[idxs[:last]])
					xx2     = np.minimum(x2[i], x2[idxs[:last]])
					yy2     = np.minimum(y2[i], y2[idxs[:last]])
					w       = np.maximum(0, xx2 - xx1 + 1)
					h       = np.maximum(0, yy2 - yy1 + 1)
					overlap = (w * h) / area[idxs[:last]]
					idxs    = np.delete(
						idxs, np.concatenate((
							[last], np.where(overlap > self.nms_max_overlap)[0]
						))
					)

			# NOTE: Get exactly the vehicles surviving non-max-suppression
			valid_inst = [inst[i] for i in indices]
			valid_instances.append(valid_inst)

		return valid_instances

	# MARK: Utils

	def is_correct_label(self, label: dict) -> bool:
		"""Check if the label is allowed in our application.

		Args:
			label (dict):
				Label dict.

		Returns:
			True or false.
		"""
		if label["train_id"] in self.allowed_ids:
			return True
		return False

	def clear_model_memory(self):
		"""Free the memory of model

		Returns:
			None
		"""
		if self.model is not None:
			del self.model
			torch.cuda.empty_cache()
