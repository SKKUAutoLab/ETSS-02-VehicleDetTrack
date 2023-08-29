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
	"BaseMatcher"
]


# MARK: - BaseMatcher
from core.utils.image import to_tensor


# noinspection PyShadowingBuiltins
class BaseMatcher(metaclass=abc.ABCMeta):
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
			name           : Optional[str] = None,
			*args, **kwargs
	):
		super().__init__()
		self.name = name

	def update(self, gmos):
		pass
