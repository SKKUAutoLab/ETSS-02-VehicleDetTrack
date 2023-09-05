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
			Name of the matcher model.
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			name           : Optional[str] = None,
			*args, **kwargs
	):
		super().__init__()
		self.name = name

	def update(self):
		pass
