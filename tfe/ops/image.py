# ==================================================================== #
# File name: image.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/27/2021
# ==================================================================== #
from typing import Tuple
from typing import Union

import cv2
import numpy as np

from tfe.utils import printe


# MARK: - Resize Ops

def letterbox(
	image      : np.ndarray,
	new_shape  : Union[int, Tuple[int, int]] = (768, 768),
	color      : Tuple[int, int, int]        = (114, 114, 114),
	auto       : bool = True,
	scale_fill : bool = False,
	scaleup    : bool = True
):
	"""
	"""
	# Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
	shape = image.shape[:2]  # current shape [height, width]
	
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)
	
	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better test mAP)
		r = min(r, 1.0)

	# Compute padding
	ratio     = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh    = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
	
	if auto:  # minimum rectangle
		dw, dh    = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
	elif scale_fill:  # stretch
		dw, dh    = 0.0, 0.0
		new_unpad = (new_shape[1], new_shape[0])
		ratio     = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

	dw /= 2  # divide padding into 2 sides
	dh /= 2
	
	if shape[::-1] != new_unpad:  # resize
		image = cv2.resize(src=image, dsize=new_unpad, interpolation=cv2.INTER_LINEAR)
		
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	image       = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	
	return image, ratio, (dw, dh)


def padded_resize_image(
	images: np.ndarray,
	size : Tuple[int, int]
) -> np.ndarray:
	"""Perform pad and resize image.
	"""
	if images.ndim == 4:
		images = [letterbox(img, new_shape=size)[0] for img in images]
	else:
		images = letterbox(images, new_shape=size)[0]

	# TODO: Convert
	# image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
	# image = image[:, :, ::-1]  # BGR to RGB, to 3x416x416
	images = np.ascontiguousarray(images)
	return images


def resize_image_cv2(
	image: np.ndarray,
	size : Tuple[int, int]
) -> np.ndarray:
	"""Resize image using OpenCV functions. Additionally, also convert image shape to channel last
	"""
	if is_channel_first(image=image):
		image = image_channel_last(image=image)
	
	if image.shape[0] != size[0] or image.shape[1] != size[1]:
		image = cv2.resize(src=image, dsize=tuple((size[1], size[0])), interpolation=cv2.INTER_LINEAR)
	
	return image
	

# MARK: - Reshape Ops

def is_channel_first(image: np.ndarray) -> bool:
	"""Return the checking position of channel value is FIRST
	"""
	if image.ndim == 3:
		if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
			return True
		elif image.shape[2] < image.shape[0] and image.shape[2] < image.shape[1]:
			return False
	elif image.ndim == 4:
		if image.shape[1] < image.shape[2] and image.shape[1] < image.shape[3]:
			return True
		elif image.shape[3] < image.shape[1] and image.shape[3] < image.shape[2]:
			return False
	else:
		printe("Image shape is not correct.")
	return False


def is_channel_last(image: np.ndarray) -> bool:
	"""Return the checking position of channel value is LAST
	"""
	return not is_channel_first(image)


def image_channel_first(image: np.ndarray) -> np.ndarray:
	"""Reshape image into shape of [CHW] or [BCHW].
	"""
	if is_channel_first(image):
		return image
	else:
		if image.ndim == 3:
			return np.transpose(image, (2, 0, 1))
		elif image.ndim == 4:
			return np.transpose(image, (0, 3, 1, 2))
		else:
			printe("Cannot reshape image. Image shape is incorrect.")
			raise ValueError


def image_channel_last(image: np.ndarray) -> np.ndarray:
	"""Reshape image into shape of [HWC] or [BHWC].
	"""
	if is_channel_last(image):
		return image
	else:
		if image.ndim == 3:
			return np.transpose(image, (1, 2, 0))
		elif image.ndim == 4:
			return np.transpose(image, (0, 2, 3, 1))
		else:
			printe("Cannot reshape image. Image shape is incorrect.")
			raise ValueError
