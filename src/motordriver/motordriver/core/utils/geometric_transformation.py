#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
https://mathworld.wolfram.com/topics/GeometricTransformations.html

List of operation:
    - Cantellation
    - Central Dilation
    - Collineation
    - Dilation
    - Elation
    - Elliptic Rotation
    - Expansion
    - Geometric Correlation
    - Geometric Homology
    - Harmonic Homology
    - Homography
    - Perspective Collineation
    - Polarity
    - Projective Collineation
    - Projective Correlation
    - Projectivity
    - Stretch
    - Twirl
    - Unimodular Transformation
"""

from __future__ import annotations

import math
import random
from enum import Enum
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Union

import cv2
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms import functional_tensor as F_t
from torchvision.transforms.functional import center_crop
from torchvision.transforms.functional import crop
from torchvision.transforms.functional import resized_crop


from core.type.type import Color
from core.type.type import Dim2
from core.type.type import Dim3
from core.type.type import ScalarOrTuple2T
from core.type.type import Size3T
from core.utils.rich import error_console
from core.utils.image import get_image_size
from core.utils.image import to_channel_last

from core.factory.builder import TRANSFORMS

__all__ = [
	"center_crop",
	"CenterCrop",
	"crop",
	"Crop",
	"interpolation_modes_from_int",
	"InterpolationMode",
	"letterbox_resize",
	"padded_resize",
	"padded_scale",
	"PaddedScale",
	"paired_random_perspective",
	"random_perspective",
	"rescale",
	"Rescale",
	"resize",
	"Resize",
	"resize_numpy_image",
	"resize_pil_image",
	"resize_tensor_image",
	"resized_crop",
	"ResizedCrop"
]


# MARK: - Interpolation Mode

class InterpolationMode(Enum):
	"""Interpolation modes. Available interpolation methods are:
	 [`nearest`, `bilinear`, `bicubic`, `box`, `hamming`, and `lanczos`].
	"""
	BICUBIC       = "bicubic"
	BILINEAR      = "bilinear"
	NEAREST       = "nearest"
	# For PIL compatibility
	BOX           = "box"
	HAMMING       = "hamming"
	LANCZOS       = "lanczos"
	# For opencv compatibility
	AREA          = "area"
	CUBIC         = "cubic"
	LANCZOS4      = "lanczos4"
	LINEAR        = "linear"
	LINEAR_EXACT  = "linear_exact"
	MAX           = "max"
	NEAREST_EXACT = "nearest_exact"


def interpolation_modes_from_int(i: int) -> InterpolationMode:
	inverse_modes_mapping = {
		0 : InterpolationMode.NEAREST,
		1 : InterpolationMode.LANCZOS,
		2 : InterpolationMode.BILINEAR,
		3 : InterpolationMode.BICUBIC,
		4 : InterpolationMode.BOX,
		5 : InterpolationMode.HAMMING,
		6 : InterpolationMode.AREA,
		7 : InterpolationMode.CUBIC,
		8 : InterpolationMode.LANCZOS4,
		9 : InterpolationMode.LINEAR,
		10: InterpolationMode.MAX,
	}
	return inverse_modes_mapping[i]


cv_modes_mapping = {
	InterpolationMode.AREA    : cv2.INTER_AREA,
	InterpolationMode.CUBIC   : cv2.INTER_CUBIC,
	InterpolationMode.LANCZOS4: cv2.INTER_LANCZOS4,
	InterpolationMode.LINEAR  : cv2.INTER_LINEAR,
	InterpolationMode.MAX     : cv2.INTER_MAX,
	InterpolationMode.NEAREST : cv2.INTER_NEAREST,
}


pil_modes_mapping = {
	InterpolationMode.NEAREST : 0,
	InterpolationMode.LANCZOS : 1,
	InterpolationMode.BILINEAR: 2,
	InterpolationMode.BICUBIC : 3,
	InterpolationMode.BOX     : 4,
	InterpolationMode.HAMMING : 5,
}


# MARK: - Crop

@TRANSFORMS.register(name="center_crop")
class CenterCrop(torch.nn.Module):
	"""Crops the given image at the center.
	If the image is Tensor, it is expected to have [..., H, W] shape,
	where ... means an arbitrary number of leading dimensions.
	If image size is smaller than output size along any edge, image is padded
	with 0 and then cropped.

	Args:
		output_size (Dim2):
			[height, width] of the crop box. If int or sequence with single int,
			it is used for both directions.
	"""

	def __init__(self, output_size: Dim2):
		super().__init__()
		self.output_size = output_size

	def forward(self, image: Tensor) -> Tensor:
		"""

		Args:
			image (PIL Image or Tensor):
				Image to be cropped.

		Returns:
			(PIL Image or Tensor):
				Cropped image.
		"""
		return center_crop(image, self.output_size)


@TRANSFORMS.register(name="crop")
class Crop(torch.nn.Module):
	"""Crop the given image at specified location and output size.
	If the image is Tensor, it is expected to have [..., H, W] shape,
	where ... means an arbitrary number of leading dimensions.
	If image size is smaller than output size along any edge, image is padded
	with 0 and then cropped.

	Args:
		top (int):
			Vertical component of the top left corner of the crop box.
		left (int):
			Horizontal component of the top left corner of the crop box.
		height (int):
			Height of the crop box.
		width (int):
			Width of the crop box.
	"""

	def __init__(self, top: int, left: int, height: int, width: int):
		super().__init__()
		self.top    = top
		self.left   = left
		self.height = height
		self.width  = width

	def forward(self, image: Tensor) -> Tensor:
		"""

		Args:
			image (PIL Image or Tensor):
				Image to be cropped. (0,0) denotes the top left corner of the
				image.

		Returns:
			(PIL Image or Tensor):
				Cropped image.
		"""
		return crop(image, self.top, self.left, self.height, self.width)


@TRANSFORMS.register(name="resized_crop")
class ResizedCrop(torch.nn.Module):
	"""Crop the given image and resize it to desired size.
	If the image is Tensor, it is expected to have [..., H, W] shape,
	where ... means an arbitrary number of leading dimensions.

	Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

	Args:
		top (int):
			Vertical component of the top left corner of the crop box.
		left (int):
			Horizontal component of the top left corner of the crop box.
		height (int):
			Height of the crop box.
		width (int):
			Width of the crop box.
		size (list[int]):
			Desired output size. Same semantics as `resize`.
		interpolation (InterpolationMode):
			Desired interpolation enum defined by
			:class:`torchvision.transforms.InterpolationMode`.
			Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
			`InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
			`InterpolationMode.BICUBIC` are supported.
			For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
			are still acceptable.
	"""

	def __init__(
			self,
			top          : int,
			left         : int,
			height       : int,
			width        : int,
			size         : list[int],
			interpolation: InterpolationMode = InterpolationMode.BILINEAR
	):
		super().__init__()
		self.top           = top
		self.left          = left
		self.height        = height
		self.width         = width
		self.size          = size
		self.interpolation = interpolation

	def forward(self, image: Tensor) -> Tensor:
		"""

		Args:
			image (PIL Image or Tensor):
				Image to be cropped. (0,0) denotes the top left corner of the
				image.

		Returns:
			(PIL Image or Tensor):
				Cropped image.
		"""
		return resized_crop(
			image, self.top, self.left, self.height, self.width, self.size,
			self.interpolation
		)


# MARK: - Homography


# MARK: - Perspective

def paired_random_perspective(
		image1     : np.ndarray,
		image2     : np.ndarray = (),
		rotate     : float      = 10,
		translate  : float      = 0.1,
		scale      : float      = 0.1,
		shear      : float      = 10,
		perspective: float      = 0.0,
		border     : Sequence   = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
	"""Perform random perspective the image and the corresponding mask.

	Args:
		image1 (np.ndarray):
			Image.
		image2 (np.ndarray):
			Fmask.
		rotate (float):
			Image rotation (+/- deg).
		translate (float):
			Image translation (+/- fraction).
		scale (float):
			Image scale (+/- gain).
		shear (float):
			Image shear (+/- deg).
		perspective (float):
			Image perspective (+/- fraction), range 0-0.001.
		border (tuple, list):

	Returns:
		image1_new (np.ndarray):
			Faugmented image.
		image2_new (np.ndarray):
			Faugmented mask.
	"""
	# torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
	# targets = [cls, xyxy]

	height     = image1.shape[0] + border[0] * 2  # Shape of [HWC]
	width      = image1.shape[1] + border[1] * 2
	image1_new = image1.copy()
	image2_new = image2.copy()

	# NOTE: Center
	C       = np.eye(3)
	C[0, 2] = -image1_new.shape[1] / 2  # x translation (pixels)
	C[1, 2] = -image1_new.shape[0] / 2  # y translation (pixels)

	# NOTE: Perspective
	P       = np.eye(3)
	# x perspective (about y)
	P[2, 0] = random.uniform(-perspective, perspective)
	# y perspective (about x)
	P[2, 1] = random.uniform(-perspective, perspective)

	# NOTE: Rotation and Scale
	R = np.eye(3)
	a = random.uniform(-rotate, rotate)
	# Add 90deg rotations to small rotations
	# a += random.choice([-180, -90, 0, 90])
	s = random.uniform(1 - scale, 1 + scale)
	# s = 2 ** random.uniform(-scale, scale)
	R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

	# NOTE: Shear
	S       = np.eye(3)
	# x shear (deg)
	S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
	# y shear (deg)
	S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

	# NOTE: Translation
	T = np.eye(3)
	# x translation (pixels)
	T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
	# y translation (pixels)
	T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

	# NOTE: Combined rotation matrix
	M = T @ S @ R @ P @ C  # Order of operations (right to left) is IMPORTANT
	# Image changed
	if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
		if perspective:
			image1_new = cv2.warpPerspective(
				image1_new, M, dsize=(width, height),
				borderValue=(114, 114, 114)
			)
			image2_new  = cv2.warpPerspective(
				image2_new, M, dsize=(width, height),
				borderValue=(114, 114, 114)
			)
		else:  # Affine
			image1_new = cv2.warpAffine(
				image1_new, M[:2], dsize=(width, height),
				borderValue=(114, 114, 114)
			)
			image2_new  = cv2.warpAffine(
				image2_new, M[:2], dsize=(width, height),
				borderValue=(114, 114, 114)
			)

	return image1_new, image2_new


def random_perspective(
		image      : np.ndarray,
		rotate     : float    = 10,
		translate  : float    = 0.1,
		scale      : float    = 0.1,
		shear      : float    = 10,
		perspective: float    = 0.0,
		border     : Sequence = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
	"""Perform random perspective the image and the corresponding mask labels.

	Args:
		image (np.ndarray):
			Image.
		rotate (float):
			Image rotation (+/- deg).
		translate (float):
			Image translation (+/- fraction).
		scale (float):
			Image scale (+/- gain).
		shear (float):
			Image shear (+/- deg).
		perspective (float):
			Image perspective (+/- fraction), range 0-0.001.
		border (tuple, list):

	Returns:
		image_new (np.ndarray):
			Faugmented image.
		mask_labels_new (np.ndarray):
			Faugmented mask.
	"""
	# torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
	# targets = [cls, xyxy]

	height    = image.shape[0] + border[0] * 2  # Shape of [HWC]
	width     = image.shape[1] + border[1] * 2
	image_new = image.copy()

	# NOTE: Center
	C       = np.eye(3)
	C[0, 2] = -image_new.shape[1] / 2  # x translation (pixels)
	C[1, 2] = -image_new.shape[0] / 2  # y translation (pixels)

	# NOTE: Perspective
	P       = np.eye(3)
	# x perspective (about y)
	P[2, 0] = random.uniform(-perspective, perspective)
	# y perspective (about x)
	P[2, 1] = random.uniform(-perspective, perspective)

	# NOTE: Rotation and Scale
	R = np.eye(3)
	a = random.uniform(-rotate, rotate)
	# Add 90deg rotations to small rotations
	# a += random.choice([-180, -90, 0, 90])
	s = random.uniform(1 - scale, 1 + scale)
	# s = 2 ** random.uniform(-scale, scale)
	R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

	# NOTE: Shear
	S       = np.eye(3)
	# x shear (deg)
	S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
	# y shear (deg)
	S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

	# NOTE: Translation
	T       = np.eye(3)
	# x translation (pixels)
	T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
	# y translation (pixels)
	T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

	# NOTE: Combined rotation matrix
	M = T @ S @ R @ P @ C  # Order of operations (right to left) is IMPORTANT
	# Image changed
	if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
		if perspective:
			image_new = cv2.warpPerspective(
				image_new, M, dsize=(width, height),
				borderValue=(114, 114, 114)
			)
		else:  # Affine
			image_new = cv2.warpAffine(
				image_new, M[:2], dsize=(width, height),
				borderValue=(114, 114, 114)
			)

	return image_new


# MARK: - Resize

def resize_tensor_image(
		image        : Tensor,
		size         : Size3T,
		interpolation: InterpolationMode = InterpolationMode.BILINEAR,
		max_size     : Optional[int]     = None,
		antialias    : Optional[bool]    = None
) -> Tensor:
	"""Resize a tensor image. Adapted from:
	`torchvision.transforms.functional_tensor.resize()`

	Args:
		image (Tensor):
			Image of shape [H, W, C].
		size (Size3T):
			Desired output size of shape [W, H, C].
		interpolation (InterpolationMode):
			Interpolation method.
		max_size (int, optional):

		antialias (bool, optional):

	Returns:
		resize (Tensor):
			Resized image of shape [H, W, C].
	"""
	if isinstance(interpolation, int):
		interpolation = interpolation_modes_from_int(interpolation)
	if not isinstance(interpolation, InterpolationMode):
		raise TypeError("Argument interpolation should be a InterpolationMode")

	if isinstance(size, (list, tuple)) and len(size) == 3:
		size = size[0:2]
	if size[0] > size[1]:
		size = (size[1], size[0])

	return F_t.resize(
		img           = image,
		size          = size,
		interpolation = interpolation.value,
		max_size      = max_size,
		antialias     = antialias
	)


def resize_numpy_image(
		image        : np.ndarray,
		size         : Size3T,
		interpolation: InterpolationMode = InterpolationMode.LINEAR,
		max_size     : Optional[int]     = None,
		antialias    : Optional[bool]    = None
) -> np.ndarray:
	"""Resize a numpy image. Adapted from:
	`torchvision.transforms.functional_tensor.resize()`

	Args:
		image (np.ndarray):
			Image of shape [H, W, C].
		size (Size3T):
			Desired output size of shape [W, H, C*].
		interpolation (InterpolationMode):
			Interpolation method.
		max_size (int, optional):

		antialias (bool, optional):

	Returns:
		resize (np.ndarray):
			Resized image of shape [H, W, C].
	"""
	if isinstance(interpolation, int):
		interpolation = interpolation_modes_from_int(interpolation)
	if not isinstance(interpolation, InterpolationMode):
		raise TypeError("Argument interpolation should be a InterpolationMode")
	cv_interpolation = cv_modes_mapping[interpolation]
	if cv_interpolation not in list(cv_modes_mapping.values()):
		raise ValueError(
			"This interpolation mode is unsupported with np.ndarray input"
		)

	if not isinstance(size, (int, tuple, list)):
		raise TypeError("Got inappropriate size arg")
	if isinstance(size, list):
		size = tuple(size)
	if len(size) == 3:
		size = size[0:2]
	if isinstance(size, tuple):
		if len(size) not in [1, 2]:
			raise ValueError(
				f"Size must be an int or a 1 or 2 element tuple/list, not a "
				f"{len(size)} element tuple/list."
			)
		if max_size is not None and len(size) != 1:
			raise ValueError(
				"max_size should only be passed if size specifies the length "
				"of the smaller edge, i.e. size should be an int or a sequence "
				"of length 1 in torchscript mode."
			)

	if antialias is None:
		antialias = False

	if antialias and cv_interpolation not in [cv2.INTER_LINEAR, cv2.INTER_CUBIC]:
		raise ValueError("Antialias option is supported for linear and cubic "
						 "interpolation modes only")

	w, h = get_image_size(image)
	# Specified size only for the smallest edge
	if isinstance(size, int) or len(size) == 1:
		short, long         = (w, h) if w <= h else (h, w)
		requested_new_short = size if isinstance(size, int) else size[0]

		if short == requested_new_short:
			return image

		new_short = requested_new_short,
		new_long  = int(requested_new_short * long / short)

		if max_size is not None:
			if max_size <= requested_new_short:
				raise ValueError(
					f"max_size = {max_size} must be strictly greater than the "
					f"requested size for the smaller edge size = {size}"
				)
			if new_long > max_size:
				new_short = int(max_size * new_short / new_long),
				new_long  = max_size

		new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)

	else:  # specified both h and w
		new_w, new_h = size[0], size[1]

	image, need_cast, need_expand, out_dtype = _cast_squeeze_in(
		image, [np.float32, np.float64]
	)

	image = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv_interpolation)

	if cv_interpolation == cv2.INTER_CUBIC and out_dtype == np.uint8:
		image = np.clip(image, 0, 255)

	image = _cast_squeeze_out(
		image, need_cast=need_cast, need_expand=need_expand, out_dtype=out_dtype
	)

	return image


def _cast_squeeze_in(
		image: np.ndarray, req_dtypes: list[Any]
) -> tuple[np.ndarray, bool, bool, Any]:
	need_expand = False
	# make image HWC
	if image.ndim == 4:
		image = np.squeeze(image, axis=0)
		need_expand = True
	image = to_channel_last(image)

	out_dtype = image.dtype
	need_cast = False
	if out_dtype not in req_dtypes:
		need_cast = True
		req_dtype = req_dtypes[0]
		image     = image.astype(req_dtype)
	return image, need_cast, need_expand, out_dtype


def _cast_squeeze_out(
		image: np.ndarray, need_cast: bool, need_expand: bool, out_dtype: Any
) -> np.ndarray:
	if need_expand:
		image = np.expand_dims(image, axis=0)

	if need_cast:
		if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
			# it is better to round before cast
			image = np.round(image)
		image = image.astype(out_dtype)

	return image


def resize_pil_image(
		image        : PIL.Image.Image,
		size         : Size3T,
		interpolation: InterpolationMode = InterpolationMode.BILINEAR,
		max_size     : Optional[int]     = None,
		antialias    : Optional[bool]    = None
) -> PIL.Image:
	"""Resize a pil image. Adapted from:
	`torchvision.transforms.functional_pil.resize()`

	Args:
		image (PIL.Image.Image):
			Image of shape [H, W, C].
		size (Size3T):
			Desired output size of shape [W, H].
		interpolation (InterpolationMode):
			Interpolation method.
		max_size (int, optional):

		antialias (bool, optional):

	Returns:
		resize (PIL.Image.Image):
			Resized image of shape [H, W, C].
	"""
	if isinstance(interpolation, int):
		interpolation = interpolation_modes_from_int(interpolation)
	if not isinstance(interpolation, InterpolationMode):
		raise TypeError("Argument interpolation should be a InterpolationMode")

	if antialias is not None and not antialias:
		error_console.log(
			"Anti-alias option is always applied for PIL Image input. "
			"Argument antialias is ignored."
		)
	pil_interpolation = pil_modes_mapping[interpolation]

	if isinstance(size, (list, tuple)) and len(size) == 3:
		size = size[0:2]

	return F_pil.resize(
		image, size=size, interpolation=pil_interpolation, max_size=max_size,
		antialias=antialias
	)


def resize(
		image        : Union[Tensor, np.ndarray, PIL.Image.Image],
		size         : Size3T,
		interpolation: InterpolationMode = InterpolationMode.LINEAR,
		max_size     : Optional[int]     = None,
		antialias    : Optional[bool]    = None
) -> Union[Tensor, np.ndarray, PIL.Image.Image]:
	"""Resize an image. Adapted from:
	`torchvision.transforms.functional.resize()`

	Args:
		image (Tensor, np.ndarray, PIL.Image.Image):
			Image of shape [H, W, C].
		size (Size3T):
			Desired output size of shape [W, H].
		interpolation (InterpolationMode):
			Interpolation method.
		max_size (int, optional):

		antialias (bool, optional):

	Returns:
		resize (Tensor, np.ndarray, PIL.Image.Image):
			Resized image of shape [H, W, C].
	"""
	if isinstance(image, Tensor):
		if interpolation is InterpolationMode.LINEAR:
			interpolation = InterpolationMode.BILINEAR
		return resize_tensor_image(image, size, interpolation, max_size, antialias)
	elif isinstance(image, np.ndarray):
		if interpolation is InterpolationMode.BILINEAR:
			interpolation = InterpolationMode.LINEAR
		return resize_numpy_image(image, size, interpolation, max_size, antialias)
	else:
		return resize_pil_image(image, size, interpolation, max_size, antialias)


"""
def cv_resize(
    image: np.ndarray, size: Size3T, interpolation: Optional[int] = None
) -> tuple[np.ndarray, tuple, tuple]:
    # NOTE: Convert to channel last to be sure
    new_image = image.copy()
    new_image = to_channel_last(new_image)
    
    # NOTE: Calculate hw0 and hw1
    h0, w0 = new_image.shape[:2]  # Original HW
    if isinstance(size, int):
        ratio  = size / max(h0, w0)  # Resize image to image_size
        h1, w1 = int(h0 * ratio), int(w0 * ratio)
    elif isinstance(size, (tuple, list)):
        ratio  = 0 if (h0 != size[0] or w0 != size[1]) else 1
        h1, w1 = size[:2]
    else:
        raise ValueError(f"Do not support new image shape of type: "
                         f"{type(size)}")
    
    if interpolation is None:
        interpolation = cv2.INTER_AREA if (ratio < 1) else cv2.INTER_LINEAR

    # NOTE: Resize
    if ratio != 1:
        # Always resize down, only resize up if training with augmentation
        new_image = cv2.resize(
            src=new_image, dsize=(w1, h1), interpolation=interpolation
        )
 
    return new_image, (h0, w0), (h1, w1)
"""


def letterbox_resize(
		image     : np.ndarray,
		new_shape : Size3T = 768,
		color     : Color  = (114, 114, 114),
		stride    : int    = 32,
		auto      : bool   = True,
		scale_fill: bool   = False,
		scale_up  : bool   = True
):
	# Resize image to a 32-pixel-multiple rectangle:
	# https://github.com/ultralytics/yolov3/issues/232
	shape = image.shape[:2]  # current shape [height, width]

	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scale_up:  # only scale down, do not scale up (for better test mAP)
		r = min(r, 1.0)

	# Compute padding
	ratio     = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh    = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

	if auto:  # minimum rectangle
		dw, dh    = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
	elif scale_fill:  # stretch
		dw, dh    = 0.0, 0.0
		new_unpad = (new_shape[1], new_shape[0])
		ratio     = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		image = cv2.resize(
			src=image, dsize=new_unpad, interpolation=cv2.INTER_LINEAR
		)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	image       = cv2.copyMakeBorder(
		image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
	)  # add border
	return image, ratio, (dw, dh)


def padded_resize(image: np.ndarray, new_shape: Dim3, stride: int) -> np.ndarray:
	"""Perform pad and resize image.

	Args:
		image (np.ndarray):
			List of image or a single one.
		new_shape (Dim3):
			Desired size as [H, W, C].

	Returns:
		image (np.ndarray):
			Converted image.
	"""
	if image.ndim == 4:
		image = [letterbox_resize(i, new_shape=new_shape, stride=stride)[0] for i in image]
		image = np.array(image)
	else:
		image = letterbox_resize(image, new_shape=new_shape, stride=stride)[0]

	# NOTE: Convert
	# image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
	# image = image[:, :, ::-1]  # BGR to RGB, to 3x416x416
	image = np.ascontiguousarray(image)
	return image


@TRANSFORMS.register(name="resize")
class Resize(torch.nn.Module):
	r"""Resize the input image to the given size.
	If the image is Tensor, it is expected to have [..., H, W] shape,
	where ... means an arbitrary number of leading dimensions.

	.. warning::
		Output image might be different depending on its type: when
		downsampling, the interpolation of PIL images and tensors is slightly
		different, because PIL applies antialiasing. This may lead to
		significant differences in the performance of a network.
		Therefore, it is preferable to train and serve a model with the same
		input types. See also below the `antialias` parameter, which can help
		making the output of PIL images and tensors closer.

	Args:
		size (Size3T):
			Desired output size. If size is a sequence like [H, W], the output
			size will be matched to this. If size is an int, the smaller edge
			of the image will be matched to this number maintaining the aspect
			ratio. i.e, if height > width, then image will be rescaled to
			:math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.
			.. note::
				In torchscript mode size as single int is not supported, use a
				sequence of length 1: `[size, ]`.
		interpolation (InterpolationMode):
			Desired interpolation enum defined by
			:class:`torchvision.transforms.InterpolationMode`.
			Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
			`InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
			`InterpolationMode.BICUBIC` are supported.
			For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
			are still acceptable.
		max_size (int, optional):
			Maximum allowed for the longer edge of the resized image: if
			the longer edge of the image is greater than `max_size` after being
			resized according to `size`, then the image is resized again so
			that the longer edge is equal to `max_size`. As a result, `size`
			might be overruled, i.e the smaller edge may be shorter than `size`.
			This is only supported if `size` is an int (or a sequence of length
			1 in torchscript mode).
		antialias (bool, optional):
			Antialias flag. If `img` is PIL Image, the flag is ignored and
			anti-alias is always used. If `img` is Tensor, the flag is False by
			default and can be set to True for `InterpolationMode.BILINEAR`
			only mode. This can help making the output for PIL images and
			tensors closer.

			.. warning::
				There is no autodiff support for `antialias=True` option with
				input `img` as Tensor.
	"""

	def __init__(
			self,
			size         : Size3T,
			interpolation: InterpolationMode = InterpolationMode.BILINEAR,
			max_size     : Optional[int]     = None,
			antialias    : Optional[bool]    = None
	):
		super().__init__()
		self.size          = size
		self.interpolation = interpolation
		self.max_size      = max_size
		self.antialias     = antialias

	def forward(
			self, image: Union[Tensor, np.ndarray, PIL.Image.Image]
	) -> Union[Tensor, np.ndarray, PIL.Image.Image]:
		"""

		Args:
			image (Tensor, np.ndarray, PIL.Image.Image):
				Image to be cropped. (0,0) denotes the top left corner of the
				image.

		Returns:
			(Tensor, np.ndarray, PIL.Image.Image):
				Resized image.
		"""
		return resize(
			image, self.size, self.interpolation, self.max_size, self.antialias,
		)


# MARK: - Scale

def padded_scale(
		image: Tensor, ratio: float = 1.0, same_shape: bool = False
) -> Tensor:
	# img(16,3,256,416), r=ratio
	# scales img(bs,3,y,x) by ratio
	if ratio == 1.0:
		return image
	else:
		h, w = image.shape[2:]
		s    = (int(h * ratio), int(w * ratio))  # new size
		img  = F.interpolate(
			image, size=s, mode="bilinear", align_corners=False
		)  # Resize
		if not same_shape:  # Pad/crop img
			gs   = 128  # 64 # 32  # (pixels) grid size
			h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
		# Value = imagenet mean
		return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def rescale(
		image        : Tensor,
		factor       : ScalarOrTuple2T[float],
		interpolation: InterpolationMode = InterpolationMode.BILINEAR,
		antialias    : bool              = False,
) -> Tensor:
	"""Rescale the input Tensor with the given factor.

	Args:
		image (Tensor):
			Ftensor image to be scale with shape of [B, C, H, W].
		factor (ScalarOrTuple2T[float]):
			Desired scaling factor in each direction. If scalar, the value is
			used for both the x- and y-direction.
		interpolation (InterpolationMode):
			Desired interpolation enum defined by
			:class:`torchvision.transforms.InterpolationMode`.
			Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
			`InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
			`InterpolationMode.BICUBIC` are supported.
			For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
			are still acceptable.
		antialias (bool, optional):
			Antialias flag. If `img` is PIL Image, the flag is ignored and
			anti-alias is always used. If `img` is Tensor, the flag is False by
			default and can be set to True for `InterpolationMode.BILINEAR`
			only mode. This can help making the output for PIL images and
			tensors closer.

	Returns:
		(Tensor):
			Frescaled image with the shape as the specified size.

	Example:
		>>> img = torch.rand(1, 3, 4, 4)
		>>> out = rescale(img, (2, 3))
		>>> print(out.shape)
		torch.Size([1, 3, 8, 12])
	"""
	if isinstance(factor, float):
		factor_vert = factor_horz = factor
	else:
		factor_vert, factor_horz  = factor

	height, width = image.size()[-2:]
	size          = [int(height * factor_vert), int(width * factor_horz)]
	return resize(
		image, size=size, interpolation=interpolation, antialias=antialias
	)


@TRANSFORMS.register(name="padded_scale")
class PaddedScale(torch.nn.Module):

	def __init__(self, ratio: float = 1.0, same_shape: bool = False):
		super().__init__()
		self.ratio      = ratio
		self.same_shape = same_shape

	def forward(self, image: Tensor) -> Tensor:
		return padded_scale(image, self.ratio, self.same_shape)


@TRANSFORMS.register(name="rescale")
class Rescale(torch.nn.Module):
	r"""Rescale the input image with the given factor.
	If the image is Tensor, it is expected to have [..., H, W] shape,
	where ... means an arbitrary number of leading dimensions.

	Args:
		factor (ScalarOrTuple2T[float]):
			Desired scaling factor in each direction. If scalar, the value is
			used for both the x- and y-direction.
		interpolation (InterpolationMode):
			Desired interpolation enum defined by
			:class:`torchvision.transforms.InterpolationMode`.
			Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
			`InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
			`InterpolationMode.BICUBIC` are supported.
			For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
			are still acceptable.
		antialias (bool, optional):
			Antialias flag. If `img` is PIL Image, the flag is ignored and
			anti-alias is always used. If `img` is Tensor, the flag is False by
			default and can be set to True for `InterpolationMode.BILINEAR`
			only mode. This can help making the output for PIL images and
			tensors closer.

			.. warning::
				There is no autodiff support for `antialias=True` option with
				input `img` as Tensor.
	"""

	def __init__(
			self,
			factor       : ScalarOrTuple2T[float],
			interpolation: InterpolationMode = InterpolationMode.BILINEAR,
			antialias    : bool              = False,
	):
		super().__init__()
		self.factor        = factor
		self.interpolation = interpolation
		self.antialias     = antialias

	def forward(self, image: Tensor) -> Tensor:
		"""

		Args:
			image (PIL Image or Tensor):
				Image to be cropped. (0,0) denotes the top left corner of the
				image.

		Returns:
			(PIL Image or Tensor):
				Rescaled image.
		"""
		return rescale(image, self.factor, self.interpolation, self.antialias)
