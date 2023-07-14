#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations for bounding box. For example: format conversion, geometric
calculations, box metrics, ...
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import cv2
import numpy as np
import torch
from multipledispatch import dispatch
from torch import Tensor
from torch.distributions import Uniform

from core.type.type import Dim2
from core.type.type import ListOrTuple2T

__all__ = [
	"batch_bbox_iou",
	"bbox_area",
	"bbox_cxcyar_to_cxcyrh",
	"bbox_cxcyar_to_cxcywh",
	"bbox_cxcyar_to_cxcywhnorm",
	"bbox_cxcyar_to_xywh",
	"bbox_cxcyar_to_xyxy",
	"bbox_cxcyrh_to_cxcyar",
	"bbox_cxcyrh_to_cxcywh",
	"bbox_cxcyrh_to_cxcywh_norm",
	"bbox_cxcyrh_to_xywh",
	"bbox_cxcyrh_to_xyxy",
	"bbox_cxcywh_norm_to_cxcyar",
	"bbox_cxcywh_norm_to_cxcyrh",
	"bbox_cxcywh_norm_to_cxcywh",
	"bbox_cxcywh_norm_to_xywh",
	"bbox_cxcywh_norm_to_xyxy",
	"bbox_cxcywh_to_cxcyar",
	"bbox_cxcywh_to_cxcywh_norm",
	"bbox_cxcywh_to_xywh",
	"bbox_cxcywh_to_xyxy",
	"bbox_generator",
	"bbox_ioa",
	"bbox_iou",
	"bbox_to_mask",
	"bbox_xywh_to_cxcyar",
	"bbox_xywh_to_cxcyrh",
	"bbox_xywh_to_cxcywh",
	"bbox_xywh_to_cxcywh_norm",
	"bbox_xywh_to_xyxy",
	"bbox_xyxy_center",
	"bbox_xyxy_to_cxcyar",
	"bbox_xyxy_to_cxcyrh",
	"bbox_xyxy_to_cxcywh",
	"bbox_xyxy_to_cxcywh_norm",
	"bbox_xyxy_to_xywh",
	"clip_bbox_xyxy",
	"cutout_bbox",
	"infer_bbox_shape",
	"is_bbox_candidates",
	"nms",
	"random_bbox_perspective",
	"scale_bbox_xyxy",
	"shift_bbox",
	"validate_bbox",
]


# MARK: - Construction


def bbox_generator(
	x_start: Tensor, y_start: Tensor, width: Tensor, height: Tensor
) -> Tensor:
	"""Generate 2D bounding boxes according to the provided start coords,
	width and height.

	Args:
		x_start (Tensor):
			A tensor containing the x coordinates of the bounding boxes to be
			extracted. Shape must be a scalar image or [B].
		y_start (Tensor):
			A tensor containing the y coordinates of the bounding boxes to be
			extracted. Shape must be a scalar image or [B].
		width (Tensor):
			Widths of the masked image. Shape must be a scalar image or [B].
		height (Tensor):
			Heights of the masked image. Shape must be a scalar image or [B].

	Returns:
		bbox (Tensor):
			Bounding box image.

	Examples:
		>>> x_start = Tensor([0, 1])
		>>> y_start = Tensor([1, 0])
		>>> width   = Tensor([5, 3])
		>>> height  = Tensor([7, 4])
		>>> bbox_generator(x_start, y_start, width, height)
		image([[[0, 1],
				 [4, 1],
				 [4, 7],
				 [0, 7]],
		<BLANKLINE>
				[[1, 0],
				 [3, 0],
				 [3, 3],
				 [1, 3]]])
	"""
	if not (x_start.shape == y_start.shape and x_start.dim() in [0, 1]):
		raise AssertionError(f"`x_start` and `y_start` must be a scalar or "
							 f"[B,]. Got {x_start}, {y_start}.")
	if not (width.shape == height.shape and width.dim() in [0, 1]):
		raise AssertionError(f"`width` and `height` must be a scalar or "
							 f"[B,]. Got {width}, {height}.")
	if not x_start.dtype == y_start.dtype == width.dtype == height.dtype:
		raise AssertionError(
			f"All tensors must be in the same dtype. Got "
			f"`x_start`({x_start.dtype}), `y_start`({x_start.dtype}), "
			f"`width`({width.dtype}), `height`({height.dtype})."
		)
	if not x_start.device == y_start.device == width.device == height.device:
		raise AssertionError(
			f"All tensors must be in the same device. Got "
			f"`x_start`({x_start.device}), `y_start`({x_start.device}), "
			f"`width`({width.device}), `height`({height.device})."
		)

	bbox = (torch.tensor(
		[[[0, 0], [0, 0], [0, 0], [0, 0]]],
		device=x_start.device, dtype=x_start.dtype)
		.repeat(1 if x_start.dim() == 0 else len(x_start), 1, 1)
	)
	bbox[:, :, 0] += x_start.view(-1, 1)
	bbox[:, :, 1] += y_start.view(-1, 1)
	bbox[:, 1, 0] += width - 1
	bbox[:, 2, 0] += width - 1
	bbox[:, 2, 1] += height - 1
	bbox[:, 3, 1] += height - 1

	return bbox


# MARK: - BBox <-> Bbox

"""Coordination of bounding box's points.

(0, 0)              Image
	  ---------------------------------- -> columns
	  |                                |
	  |        ----- -> x              |
	  |        |   |                   |
	  |        |   |                   |
	  |        -----                   |
	  |        |                       |
	  |        V                       |
	  |        y                       |
	  ----------------------------------
	  |                                 (n, m)
	  V
	 rows
"""


@dispatch(Tensor)
def bbox_cxcyar_to_cxcyrh(cxcyar: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- `area` is `width * height`.
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcyar.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		width     = torch.sqrt(cxcyar[2] * cxcyar[3])
		height    = cxcyar[2] / width
		cxcyrh[2] = cxcyar[3]
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = torch.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights      = cxcyar[:, 2] / widths
		cxcyrh[:, 2] = cxcyar[:, 3]
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_cxcyar_to_cxcyrh(cxcyar: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- `area` is `width * height`.
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcyar.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		width     = np.sqrt(cxcyar[2] * cxcyar[3])
		height    = cxcyar[2] / width
		cxcyrh[2] = cxcyar[3]
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = np.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights      = cxcyar[:, 2] / widths
		cxcyrh[:, 2] = cxcyar[:, 3]
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(Tensor)
def bbox_cxcyar_to_cxcywh(cxcyar: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[center_x, center_y, width, height].
	Where:
		- `area` is `width * height`.
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcyar.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		width     = torch.sqrt(cxcyar[2] * cxcyar[3])
		height    = cxcyar[2] / width
		cxcyrh[2] = width
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = torch.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights      = cxcyar[:, 2] / widths
		cxcyrh[:, 2] = widths
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_cxcyar_to_cxcywh(cxcyar: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[center_x, center_y, width, height].
	Where:
		- `area` is `width * height`.
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcyar.copy()
	cxcyrh = cxcyrh.astype(np.float32)
	if cxcyrh.ndim == 1:
		width     = np.sqrt(cxcyar[2] * cxcyar[3])
		height    = cxcyar[2] / width
		cxcyrh[2] = width
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = np.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights      = cxcyar[:, 2] / widths
		cxcyrh[:, 2] = widths
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(Tensor, (int, float), (int, float))
def bbox_cxcyar_to_cxcywhnorm(
	cxcyar: Tensor, height, width
) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- `area` is `width * height`.
		- `aspect_ratio` is `width / height`.
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcyrh = cxcyar.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		w         = torch.sqrt(cxcyar[2] * cxcyar[3])
		h         = cxcyar[2] / w
		cxcyrh[0] /= width
		cxcyrh[1] /= height
		cxcyrh[2] = (w / width)
		cxcyrh[3] = (h / width)
	elif cxcyrh.ndim == 2:
		ws           = torch.sqrt(cxcyar[:, 2] * cxcyar[: , 3])
		hs           = cxcyar[:, 2] / ws
		cxcyrh[:, 0] = cxcyrh[:, 0] / width
		cxcyrh[:, 1] = cxcyrh[:, 1] / height
		cxcyrh[:, 2] = (ws / width)
		cxcyrh[:, 3] = (hs / width)
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcyar_to_cxcywhnorm(cxcyar: np.ndarray, height, width) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- `area` is `width * height`.
		- `aspect_ratio` is `width / height`.
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcyrh = cxcyar.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		w = np.sqrt(cxcyar[2] * cxcyar[3])
		h = cxcyar[2] / w
		cxcyrh[0] /= width
		cxcyrh[1] /= height
		cxcyrh[2] = (w / width)
		cxcyrh[3] = (h / width)
	elif cxcyrh.ndim == 2:
		ws = np.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		hs = cxcyar[:, 2] / ws
		cxcyrh[:, 0] = cxcyrh[:, 0] / width
		cxcyrh[:, 1] = cxcyrh[:, 1] / height
		cxcyrh[:, 2] = (ws / width)
		cxcyrh[:, 3] = (hs / width)
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(Tensor)
def bbox_cxcyar_to_xywh(cxcyar: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[top_left_x, top_left_y, width, height].
	Where:
		- `area` is `width * height`.
		- `aspect_ratio` is `width / height`.
	"""
	xywh = cxcyar.clone()
	xywh = xywh.float()
	if xywh.ndim == 1:
		width   = torch.sqrt(cxcyar[2] * cxcyar[3])
		height  = cxcyar[2] / width
		xywh[0] = xywh[0] - (width / 2.0)
		xywh[1] = xywh[1] - (height / 2.0)
		xywh[2] = width
		xywh[3] = height
	elif xywh.ndim == 2:
		widths     = torch.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights    = cxcyar[:, 2] / widths
		xywh[:, 0] = xywh[:, 0] - (widths / 2.0)
		xywh[:, 1] = xywh[:, 1] - (heights / 2.0)
		xywh[:, 2] = widths
		xywh[:, 3] = heights
	else:
		raise ValueError(f"Farray dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(np.ndarray)
def bbox_cxcyar_to_xywh(cxcyar: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[top_left_x, top_left_y, width, height].
	Where:
		- `area` is `width * height`.
		- `aspect_ratio` is `width / height`.
	"""
	xywh = cxcyar.copy()
	xywh = xywh.astype(float)
	if xywh.ndim == 1:
		width   = np.sqrt(cxcyar[2] * cxcyar[3])
		height  = cxcyar[2] / width
		xywh[0] = xywh[0] - (width / 2.0)
		xywh[1] = xywh[1] - (height / 2.0)
		xywh[2] = width
		xywh[3] = height
	elif xywh.ndim == 2:
		widths     = np.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights    = cxcyar[:, 2] / widths
		xywh[:, 0] = xywh[:, 0] - (widths / 2.0)
		xywh[:, 1] = xywh[:, 1] - (heights / 2.0)
		xywh[:, 2] = widths
		xywh[:, 3] = heights
	else:
		raise ValueError(f"Farray dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(Tensor)
def bbox_cxcyar_to_xyxy(cxcyar: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	Where:
		- `area` is `width * height`.
		- `aspect_ratio` is `width / height`.
	"""
	xyxy = cxcyar.clone()
	xyxy = xyxy.float()
	if xyxy.ndim == 1:
		width   = torch.sqrt(cxcyar[2] * cxcyar[3])
		height  = cxcyar[2] / width
		xyxy[0] = xyxy[0] - (width / 2.0)
		xyxy[1] = xyxy[1] - (height / 2.0)
		xyxy[2] = xyxy[2] + (width / 2.0)
		xyxy[3] = xyxy[3] + (height / 2.0)
	elif xyxy.ndim == 2:
		widths     = torch.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights    = cxcyar[:, 2] / widths
		xyxy[:, 0] = xyxy[:, 0] - (widths / 2.0)
		xyxy[:, 1] = xyxy[:, 1] - (heights / 2.0)
		xyxy[:, 2] = xyxy[:, 2] + (widths / 2.0)
		xyxy[:, 3] = xyxy[:, 3] + (heights / 2.0)
	else:
		raise ValueError(f"Farray dimensions {xyxy.ndim} is not supported.")
	return xyxy


@dispatch(np.ndarray)
def bbox_cxcyar_to_xyxy(cxcyar: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	Where:
		- `area` is `width * height`.
		- `aspect_ratio` is `width / height`.
	"""
	xyxy = cxcyar.copy()
	xyxy = xyxy.astype(float)
	if xyxy.ndim == 1:
		width   = np.sqrt(cxcyar[2] * cxcyar[3])
		height  = cxcyar[2] / width
		xyxy[0] = xyxy[0] - (width / 2.0)
		xyxy[1] = xyxy[1] - (height / 2.0)
		xyxy[2] = xyxy[2] + (width / 2.0)
		xyxy[3] = xyxy[3] + (height / 2.0)
	elif xyxy.ndim == 2:
		widths     = np.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights    = cxcyar[:, 2] / widths
		xyxy[:, 0] = xyxy[:, 0] - (widths / 2.0)
		xyxy[:, 1] = xyxy[:, 1] - (heights / 2.0)
		xyxy[:, 2] = xyxy[:, 2] + (widths / 2.0)
		xyxy[:, 3] = xyxy[:, 3] + (heights / 2.0)
	else:
		raise ValueError(f"Farray dimensions {xyxy.ndim} is not supported.")
	return xyxy


@dispatch(Tensor)
def bbox_cxcyrh_to_cxcyar(cxcyrh: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- `area` is `width * height`.
		- `aspect_ratio` is `width / height`.
	"""
	cxcyar = cxcyrh.clone()
	cxcyar = cxcyar.float()
	if cxcyar.ndim == 1:
		width     = cxcyrh[2] * cxcyrh[3]
		height    = cxcyrh[3]
		cxcyar[2] = width * height
		cxcyar[3] = width / height
	elif cxcyar.ndim == 2:
		widths       = cxcyrh[:, 2] * cxcyrh[:, 3]
		heights      = cxcyrh[:, 3]
		cxcyar[:, 2] = widths * heights
		cxcyar[:, 3] = widths / heights
	else:
		raise ValueError(f"Farray dimensions {cxcyar.ndim} is not "
						 f"supported.")
	return cxcyar


@dispatch(np.ndarray)
def bbox_cxcyrh_to_cxcyar(cxcyrh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- `area` is `width * height`.
		- `aspect_ratio` is `width / height`.
	"""
	cxcyar = cxcyrh.copy()
	cxcyar = cxcyar.astype(float)
	if cxcyar.ndim == 1:
		width     = cxcyrh[2] * cxcyrh[3]
		height    = cxcyrh[3]
		cxcyar[2] = width * height
		cxcyar[3] = width / height
	elif cxcyar.ndim == 2:
		widths       = cxcyrh[:, 2] * cxcyrh[:, 3]
		heights      = cxcyrh[:, 3]
		cxcyar[:, 2] = widths * heights
		cxcyar[:, 3] = widths / heights
	else:
		raise ValueError(f"Farray dimensions {cxcyar.ndim} is not "
						 f"supported.")
	return cxcyar


@dispatch(Tensor)
def bbox_cxcyrh_to_cxcywh(cxcyrh: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[center_x, center_y, width, height].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcywh = cxcyrh.clone()
	cxcywh = cxcywh.float()
	if cxcywh.ndim == 1:
		cxcywh[2] = cxcywh[2] * cxcywh[3]
	elif cxcywh.ndim == 2:
		cxcywh[:, 2] = cxcywh[:, 2] * cxcywh[:, 3]
	else:
		raise ValueError(f"Farray dimensions {cxcywh.ndim} is not "
						 f"supported.")
	return cxcywh


@dispatch(np.ndarray)
def bbox_cxcyrh_to_cxcywh(cxcyrh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[center_x, center_y, width, height].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcywh = cxcyrh.copy()
	cxcywh = cxcywh.astype(float)
	if cxcywh.ndim == 1:
		cxcywh[2] = cxcywh[2] * cxcywh[3]
	elif cxcywh.ndim == 2:
		cxcywh[:, 2] = cxcywh[:, 2] * cxcywh[:, 3]
	else:
		raise ValueError(f"Farray dimensions {cxcywh.ndim} is not "
						 f"supported.")
	return cxcywh


@dispatch(Tensor, (int, float), (int, float))
def bbox_cxcyrh_to_cxcywh_norm(
	cxcyrh: Tensor, height, width
) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- `aspect_ratio` is `width / height`.
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_cxcyrh_to_cxcywh(cxcyrh)
	if cxcyrh.ndim == 1:
		cxcywh_norm[0] /= width
		cxcywh_norm[1] /= height
		cxcywh_norm[2] /= width
		cxcywh_norm[3] /= height
	elif cxcyrh.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcyrh_to_cxcywh_norm(cxcyrh: np.ndarray, height, width) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- `aspect_ratio` is `width / height`.
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_cxcyrh_to_cxcywh(cxcyrh)
	if cxcyrh.ndim == 1:
		cxcywh_norm[0] /= width
		cxcywh_norm[1] /= height
		cxcywh_norm[2] /= width
		cxcywh_norm[3] /= height
	elif cxcyrh.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(Tensor)
def bbox_cxcyrh_to_xywh(cxcyrh: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[top_left_x, top_left_y, width, height].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	xywh = cxcyrh.clone()
	xywh = xywh.float()
	if xywh.ndim == 1:
		width   = xywh[2] * xywh[3]
		height  = xywh[3]
		xywh[0] = xywh[0] - width / 2.0
		xywh[1] = xywh[1] - height / 2.0
		xywh[2] = width
		xywh[3] = height
	elif xywh.ndim == 2:
		widths     = xywh[:, 2] * xywh[:, 3]
		heights    = xywh[:, 3]
		xywh[:, 0] = xywh[:, 0] - widths / 2.0
		xywh[:, 1] = xywh[:, 1] - heights / 2.0
		xywh[:, 2] = widths
		xywh[:, 3] = heights
	else:
		raise ValueError(f"Farray dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(np.ndarray)
def bbox_cxcyrh_to_xywh(cxcyrh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[top_left_x, top_left_y, width, height].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	xywh = cxcyrh.copy()
	xywh = xywh.astype(float)
	if xywh.ndim == 1:
		width   = xywh[2] * xywh[3]
		height  = xywh[3]
		xywh[0] = xywh[0] - width / 2.0
		xywh[1] = xywh[1] - height / 2.0
		xywh[2] = width
		xywh[3] = height
	elif xywh.ndim == 2:
		widths     = xywh[:, 2] * xywh[:, 3]
		heights    = xywh[:, 3]
		xywh[:, 0] = xywh[:, 0] - widths  / 2.0
		xywh[:, 1] = xywh[:, 1] - heights / 2.0
		xywh[:, 2] = widths
		xywh[:, 3] = heights
	else:
		raise ValueError(f"Farray dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(Tensor)
def bbox_cxcyrh_to_xyxy(cxcyrh: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y],
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	xyxy = cxcyrh.clone()
	xyxy = xyxy.float()
	if xyxy.ndim == 1:
		half_height = xyxy[3] / 2.0
		half_width  = (xyxy[2] * xyxy[3]) / 2.0
		xyxy[3]     = xyxy[3] + half_height
		xyxy[2]     = xyxy[2] + half_width
		xyxy[1]     = xyxy[1] - half_height
		xyxy[0]     = xyxy[0] - half_width
	elif xyxy.ndim == 2:
		half_heights = xyxy[:, 3] / 2.0
		half_widths  = (xyxy[:, 2] * xyxy[:, 3]) / 2.0
		xyxy[:, 3]   = xyxy[:, 3] + half_heights
		xyxy[:, 2]   = xyxy[:, 2] + half_widths
		xyxy[:, 1]   = xyxy[:, 1] - half_heights
		xyxy[:, 0]   = xyxy[:, 0] - half_widths
	else:
		raise ValueError(f"Farray dimensions {xyxy.ndim} is not supported.")
	return xyxy


@dispatch(np.ndarray)
def bbox_cxcyrh_to_xyxy(cxcyrh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	xyxy = cxcyrh.copy()
	xyxy = xyxy.astype(float)
	if xyxy.ndim == 1:
		half_height = xyxy[3] / 2.0
		half_width  = (xyxy[2] * xyxy[3]) / 2.0
		xyxy[3]     = xyxy[3] + half_height
		xyxy[2]     = xyxy[2] + half_width
		xyxy[1]     = xyxy[1] - half_height
		xyxy[0]     = xyxy[0] - half_width
	elif xyxy.ndim == 2:
		half_heights = xyxy[:, 3] / 2.0
		half_widths  = (xyxy[:, 2] * xyxy[:, 3]) / 2.0
		xyxy[:, 3]   = xyxy[:, 3] + half_heights
		xyxy[:, 2]   = xyxy[:, 2] + half_widths
		xyxy[:, 1]   = xyxy[:, 1] - half_heights
		xyxy[:, 0]   = xyxy[:, 0] - half_widths
	else:
		raise ValueError(f"Farray dimensions {xyxy.ndim} is not supported.")
	return xyxy


@dispatch(Tensor)
def bbox_cxcywh_to_cxcyar(cxcywh: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcywh.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		cxcyrh[2] = cxcywh[2] * cxcywh[3]
		cxcyrh[3] = cxcywh[2] / cxcywh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 2] = cxcywh[:, 2] * cxcywh[:, 3]
		cxcyrh[:, 3] = cxcywh[:, 2] / cxcywh[:, 3]
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_cxcywh_to_cxcyar(cxcywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[center_x, center_y, aspect_ratio, height]
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcywh.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		cxcyrh[2] = cxcyrh[2] / cxcyrh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 2] = cxcyrh[:, 2] / cxcyrh[:, 3]
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(Tensor)
def bbox_cxcywh_to_cxcyar(cxcywh: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[center_x, center_y, aspect_ratio, height]
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcywh.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		cxcyrh[2] = cxcyrh[2] / cxcyrh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 2] = cxcyrh[:, 2] / cxcyrh[:, 3]
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_cxcywh_to_cxcyar(cxcywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcywh.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		cxcyrh[2] = cxcywh[2] * cxcywh[3]
		cxcyrh[3] = cxcywh[2] / cxcywh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 2] = cxcywh[:, 2] * cxcywh[:, 3]
		cxcyrh[:, 3] = cxcywh[:, 2] / cxcywh[:, 3]
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(Tensor, (int, float), (int, float))
def bbox_cxcywh_to_cxcywh_norm(
	cxcywh: Tensor, height, width
) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = cxcywh.clone()
	cxcywh_norm = cxcywh_norm.float()
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] /= width
		cxcywh_norm[1] /= height
		cxcywh_norm[2] /= width
		cxcywh_norm[3] /= height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"Farray dimensions {cxcywh_norm.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcywh_to_cxcywh_norm(
	cxcywh: np.ndarray, height, width
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = cxcywh.copy()
	cxcywh_norm = cxcywh_norm.astype(float)
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] /= width
		cxcywh_norm[1] /= height
		cxcywh_norm[2] /= width
		cxcywh_norm[3] /= height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"Farray dimensions {cxcywh_norm.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(Tensor)
def bbox_cxcywh_to_xywh(cxcywh: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[top_left_x, top_left_y, width, height].
	"""
	xywh = cxcywh.clone()
	xywh = xywh.float()
	if xywh.ndim == 1:
		xywh[0] = xywh[0] - xywh[2] / 2.0
		xywh[1] = xywh[1] - xywh[3] / 2.0
	elif xywh.ndim == 2:
		xywh[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
		xywh[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
	else:
		raise ValueError(f"Farray dimensions {xywh.ndim} is not "
						 f"supported.")
	return xywh


@dispatch(np.ndarray)
def bbox_cxcywh_to_xywh(cxcywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[top_left_x, top_left_y, width, height].
	"""
	xywh = cxcywh.copy()
	xywh = xywh.astype(float)
	if xywh.ndim == 1:
		xywh[0] = xywh[0] - xywh[2] / 2.0
		xywh[1] = xywh[1] - xywh[3] / 2.0
	elif xywh.ndim == 2:
		xywh[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
		xywh[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
	else:
		raise ValueError(f"Farray dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(Tensor)
def bbox_cxcywh_to_xyxy(cxcywh: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	xyxy = cxcywh.clone()
	xyxy = xyxy.float()
	if xyxy.ndim == 1:
		half_width  = xyxy[2] / 2.0
		half_height = xyxy[3] / 2.0
		xyxy[3]     = xyxy[3] + half_height
		xyxy[2]     = xyxy[2] + half_width
		xyxy[1]     = xyxy[1] - half_height
		xyxy[0]     = xyxy[0] - half_width
	elif xyxy.ndim == 2:
		half_widths  = xyxy[:, 2] / 2.0
		half_heights = xyxy[:, 3] / 2.0
		xyxy[:, 3]   = xyxy[:, 3] + half_heights
		xyxy[:, 2]   = xyxy[:, 2] + half_widths
		xyxy[:, 1]   = xyxy[:, 1] - half_heights
		xyxy[:, 0]   = xyxy[:, 0] - half_widths
	else:
		raise ValueError(f"Farray dimensions {xyxy.ndim} is not "
						 f"supported.")
	return xyxy


@dispatch(np.ndarray)
def bbox_cxcywh_to_xyxy(cxcywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	xyxy = cxcywh.copy()
	xyxy = xyxy.astype(float)
	if xyxy.ndim == 1:
		half_width  = xyxy[2] / 2.0
		half_height = xyxy[3] / 2.0
		xyxy[3]     = xyxy[3] + half_height
		xyxy[2]     = xyxy[2] + half_width
		xyxy[1]     = xyxy[1] - half_height
		xyxy[0]     = xyxy[0] - half_width
	elif xyxy.ndim == 2:
		half_widths  = xyxy[:, 2] / 2.0
		half_heights = xyxy[:, 3] / 2.0
		xyxy[:, 3]   = xyxy[:, 3] + half_heights
		xyxy[:, 2]   = xyxy[:, 2] + half_widths
		xyxy[:, 1]   = xyxy[:, 1] - half_heights
		xyxy[:, 0]   = xyxy[:, 0] - half_widths
	else:
		raise ValueError(f"Farray dimensions {xyxy.ndim} is not "
						 f"supported.")
	return xyxy


@dispatch(Tensor, (int, float), (int, float))
def bbox_cxcywh_norm_to_cxcyar(
	cxcywh_norm: Tensor, height, width
) -> Tensor:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- `aspect_ratio` is `width / height`.
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcyrh = cxcywh_norm.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		cxcyrh[0] *= width
		cxcyrh[1] *= height
		cxcyrh[2] = (cxcywh_norm[2] * width) * (cxcywh_norm[3] * height)
		cxcyrh[3] = (cxcywh_norm[2] * width) / (cxcywh_norm[3] * height)
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] * width
		cxcyrh[:, 1] = cxcyrh[:, 1] * height
		cxcyrh[:, 2] = (cxcywh_norm[:, 2] * width) * (cxcywh_norm[:, 3] * height)
		cxcyrh[:, 3] = (cxcywh_norm[:, 2] * width) / (cxcywh_norm[:, 3] * height)
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcywh_norm_to_cxcyar(
	cxcywh_norm: np.ndarray, height, width
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- `aspect_ratio` is `width / height`.
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcyrh = cxcywh_norm.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		cxcyrh[0] *= width
		cxcyrh[1] *= height
		cxcyrh[2] = (cxcywh_norm[2] * width) * (cxcywh_norm[3] * height)
		cxcyrh[3] = (cxcywh_norm[2] * width) / (cxcywh_norm[3] * height)
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] * width
		cxcyrh[:, 1] = cxcyrh[:, 1] * height
		cxcyrh[:, 2] = (cxcywh_norm[:, 2] * width) * (cxcywh_norm[:, 3] * height)
		cxcyrh[:, 3] = (cxcywh_norm[:, 2] * width) / (cxcywh_norm[:, 3] * height)
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(Tensor, (int, float), (int, float))
def bbox_cxcywh_norm_to_cxcyrh(
	cxcywh_norm: Tensor, height, width
) -> Tensor:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- `aspect_ratio` is `width / height`.
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcyrh = cxcywh_norm.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		cxcyrh[0] *= width
		cxcyrh[1] *= height
		cxcyrh[3] *= height
		cxcyrh[2] = (cxcyrh[2] * width) / cxcyrh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] * width
		cxcyrh[:, 1] = cxcyrh[:, 1] * height
		cxcyrh[:, 3] = cxcyrh[:, 3] * height
		cxcyrh[:, 2] = (cxcyrh[:, 2] * width) / cxcyrh[:, 3]
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcywh_norm_to_cxcyrh(
	cxcywh_norm: np.ndarray, height, width
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- `aspect_ratio` is `width / height`.
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcyrh = cxcywh_norm.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		cxcyrh[0] *= width
		cxcyrh[1] *= height
		cxcyrh[3] *= height
		cxcyrh[2] = (cxcyrh[2] * width) / cxcyrh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] * width
		cxcyrh[:, 1] = cxcyrh[:, 1] * height
		cxcyrh[:, 3] = cxcyrh[:, 3] * height
		cxcyrh[:, 2] = (cxcyrh[:, 2] * width) / cxcyrh[:, 3]
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(Tensor, (int, float), (int, float))
def bbox_cxcywh_norm_to_cxcywh(
	cxcywh_norm: Tensor, height, width
) -> Tensor:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[center_x, center_y, width, height].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh = cxcywh_norm.clone()
	cxcywh = cxcywh.float()
	if cxcywh.ndim == 1:
		cxcywh[0] *= width
		cxcywh[1] *= height
		cxcywh[2] *= width
		cxcywh[3] *= height
	elif cxcywh.ndim == 2:
		cxcywh[:, 0] = cxcywh[:, 0] * width
		cxcywh[:, 1] = cxcywh[:, 1] * height
		cxcywh[:, 2] = cxcywh[:, 2] * width
		cxcywh[:, 3] = cxcywh[:, 3] * height
	else:
		raise ValueError(f"Farray dimensions {cxcywh.ndim} is not "
						 f"supported.")
	return cxcywh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcywh_norm_to_cxcywh(
	cxcywh_norm: np.ndarray, height, width
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[center_x, center_y, width, height].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh = cxcywh_norm.copy()
	cxcywh = cxcywh.astype(float)
	if cxcywh.ndim == 1:
		cxcywh[0] *= width
		cxcywh[1] *= height
		cxcywh[2] *= width
		cxcywh[3] *= height
	elif cxcywh.ndim == 2:
		cxcywh[:, 0] = cxcywh[:, 0] * width
		cxcywh[:, 1] = cxcywh[:, 1] * height
		cxcywh[:, 2] = cxcywh[:, 2] * width
		cxcywh[:, 3] = cxcywh[:, 3] * height
	else:
		raise ValueError(f"Farray dimensions {cxcywh.ndim} is not "
						 f"supported.")
	return cxcywh


@dispatch(Tensor, (int, float), (int, float))
def bbox_cxcywh_norm_to_xywh(
	cxcywh_norm: Tensor, height, width
) -> Tensor:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[top_left_x, top_left_y, width, height].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	xywh = cxcywh_norm.clone()
	xywh = xywh.float()
	if xywh.ndim == 1:
		xywh[3] *= height
		xywh[2] *= width
		xywh[1] = (xywh[1] * height) - (xywh[3] / 2.0)
		xywh[0] = (xywh[0] * width) - (xywh[2] / 2.0)
	elif xywh.ndim == 2:
		xywh[:, 3] = xywh[:, 3] * height
		xywh[:, 2] = xywh[:, 2] * width
		xywh[:, 1] = (xywh[:, 1] * height) - (xywh[:, 3] / 2.0)
		xywh[:, 0] = (xywh[:, 0] * width) - (xywh[:, 2] / 2.0)
	else:
		raise ValueError(f"Farray dimensions {xywh.ndim} is not "
						 f"supported.")
	return xywh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcywh_norm_to_xywh(
	cxcywh_norm: np.ndarray, height, width
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[top_left_x, top_left_y, width, height].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	xywh = cxcywh_norm.copy()
	xywh = xywh.astype(float)
	if xywh.ndim == 1:
		xywh[3] *= height
		xywh[2] *= width
		xywh[1] = (xywh[1] * height) - (xywh[3] / 2.0)
		xywh[0] = (xywh[0] * width) - (xywh[2] / 2.0)
	elif xywh.ndim == 2:
		xywh[:, 3] = xywh[:, 3] * height
		xywh[:, 2] = xywh[:, 2] * width
		xywh[:, 1] = (xywh[:, 1] * height) - (xywh[:, 3] / 2.0)
		xywh[:, 0] = (xywh[:, 0] * width) - (xywh[:, 2] / 2.0)
	else:
		raise ValueError(f"Farray dimensions {xywh.ndim} is not "
						 f"supported.")
	return xywh


@dispatch(Tensor, (int, float), (int, float))
def bbox_cxcywh_norm_to_xyxy(
	cxcywh_norm: Tensor, height, width
) -> Tensor:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	xyxy = cxcywh_norm.clone()
	xyxy = xyxy.float()
	if xyxy.ndim == 1:
		xyxy[0] = width  * (cxcywh_norm[0] - cxcywh_norm[2] / 2)
		xyxy[1] = height * (cxcywh_norm[1] - cxcywh_norm[3] / 2)
		xyxy[2] = width  * (cxcywh_norm[0] + cxcywh_norm[2] / 2)
		xyxy[3] = height * (cxcywh_norm[1] + cxcywh_norm[3] / 2)
	elif xyxy.ndim == 2:
		xyxy[:, 0] = width  * (cxcywh_norm[:, 0] - cxcywh_norm[:, 2] / 2)
		xyxy[:, 1] = height * (cxcywh_norm[:, 1] - cxcywh_norm[:, 3] / 2)
		xyxy[:, 2] = width  * (cxcywh_norm[:, 0] + cxcywh_norm[:, 2] / 2)
		xyxy[:, 3] = height * (cxcywh_norm[:, 1] + cxcywh_norm[:, 3] / 2)
	else:
		raise ValueError(f"Farray dimensions {xyxy.ndim} is not "
						 f"supported.")
	return xyxy


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcywh_norm_to_xyxy(
	cxcywh_norm: np.ndarray, height, width
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	xyxy = cxcywh_norm.copy()
	xyxy = xyxy.astype(float)
	if xyxy.ndim == 1:
		xyxy[0] = width  * (cxcywh_norm[0] - cxcywh_norm[2] / 2)
		xyxy[1] = height * (cxcywh_norm[1] - cxcywh_norm[3] / 2)
		xyxy[2] = width  * (cxcywh_norm[0] + cxcywh_norm[2] / 2)
		xyxy[3] = height * (cxcywh_norm[1] + cxcywh_norm[3] / 2)
	elif xyxy.ndim == 2:
		xyxy[:, 0] = width  * (cxcywh_norm[:, 0] - cxcywh_norm[:, 2] / 2)
		xyxy[:, 1] = height * (cxcywh_norm[:, 1] - cxcywh_norm[:, 3] / 2)
		xyxy[:, 2] = width  * (cxcywh_norm[:, 0] + cxcywh_norm[:, 2] / 2)
		xyxy[:, 3] = height * (cxcywh_norm[:, 1] + cxcywh_norm[:, 3] / 2)
	else:
		raise ValueError(f"Farray dimensions {xyxy.ndim} is not "
						 f"supported.")
	return xyxy


@dispatch(Tensor)
def bbox_xywh_to_cxcyar(xywh: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, area, aspect_ratio]
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xywh.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		cxcyrh[0] = cxcyrh[0] + (cxcyrh[2] / 2.0)
		cxcyrh[1] = cxcyrh[1] + (cxcyrh[3] / 2.0)
		cxcyrh[2] = xywh[2] * xywh[3]
		cxcyrh[3] = xywh[2] / xywh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] + (cxcyrh[:, 2] / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (cxcyrh[:, 3] / 2.0)
		cxcyrh[:, 2] = xywh[:, 2] * xywh[:, 3]
		cxcyrh[:, 3] = xywh[:, 2] / xywh[:, 3]
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_xywh_to_cxcyar(xywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xywh.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		cxcyrh[0] = cxcyrh[0] + (cxcyrh[2] / 2.0)
		cxcyrh[1] = cxcyrh[1] + (cxcyrh[3] / 2.0)
		cxcyrh[2] = xywh[2] * xywh[3]
		cxcyrh[3] = xywh[2] / xywh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] + (cxcyrh[:, 2] / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (cxcyrh[:, 3] / 2.0)
		cxcyrh[:, 2] = xywh[:, 2] * xywh[:, 3]
		cxcyrh[:, 3] = xywh[:, 2] / xywh[:, 3]
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(Tensor)
def bbox_xywh_to_cxcyrh(xywh: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xywh.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		cxcyrh[0] = cxcyrh[0] + (cxcyrh[2] / 2.0)
		cxcyrh[1] = cxcyrh[1] + (cxcyrh[3] / 2.0)
		cxcyrh[2] = cxcyrh[2] / cxcyrh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] + (cxcyrh[:, 2] / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (cxcyrh[:, 3] / 2.0)
		cxcyrh[:, 2] = cxcyrh[:, 2] / cxcyrh[:, 3]
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_xywh_to_cxcyrh(xywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xywh.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		cxcyrh[0] = cxcyrh[0] + (cxcyrh[2] / 2.0)
		cxcyrh[1] = cxcyrh[1] + (cxcyrh[3] / 2.0)
		cxcyrh[2] = cxcyrh[2] / cxcyrh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] + (cxcyrh[:, 2] / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (cxcyrh[:, 3] / 2.0)
		cxcyrh[:, 2] = cxcyrh[:, 2] / cxcyrh[:, 3]
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(Tensor)
def bbox_xywh_to_cxcywh(xywh: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, width, height].
	"""
	cxcywh = xywh.clone()
	cxcywh = cxcywh.float()
	if cxcywh.ndim == 1:
		cxcywh[0]    = cxcywh[0] + (cxcywh[2] / 2.0)
		cxcywh[1]    = cxcywh[1] + (cxcywh[3] / 2.0)
	elif cxcywh.ndim == 2:
		cxcywh[:, 0] = cxcywh[:, 0] + (cxcywh[:, 2] / 2.0)
		cxcywh[:, 1] = cxcywh[:, 1] + (cxcywh[:, 3] / 2.0)
	else:
		raise ValueError(f"Farray dimensions {cxcywh.ndim} is not "
						 f"supported.")
	return cxcywh


@dispatch(np.ndarray)
def bbox_xywh_to_cxcywh(xywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, width, height].
	"""
	cxcywh = xywh.copy()
	cxcywh = cxcywh.astype(float)
	if cxcywh.ndim == 1:
		cxcywh[0]    = cxcywh[0] + (cxcywh[2] / 2.0)
		cxcywh[1]    = cxcywh[1] + (cxcywh[3] / 2.0)
	elif cxcywh.ndim == 2:
		cxcywh[:, 0] = cxcywh[:, 0] + (cxcywh[:, 2] / 2.0)
		cxcywh[:, 1] = cxcywh[:, 1] + (cxcywh[:, 3] / 2.0)
	else:
		raise ValueError(f"Farray dimensions {cxcywh.ndim} is not "
						 f"supported.")
	return cxcywh


@dispatch(Tensor, (int, float), (int, float))
def bbox_xywh_to_cxcywh_norm(xywh: Tensor, height, width) -> Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_xywh_to_cxcywh(xywh)
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] /= width
		cxcywh_norm[1] /= height
		cxcywh_norm[2] /= width
		cxcywh_norm[3] /= height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"Farray dimensions {cxcywh_norm.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_xywh_to_cxcywh_norm(xywh: np.ndarray, height, width) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_xywh_to_cxcywh(xywh)
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] /= width
		cxcywh_norm[1] /= height
		cxcywh_norm[2] /= width
		cxcywh_norm[3] /= height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"Farray dimensions {cxcywh_norm.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(Tensor)
def bbox_xywh_to_xyxy(xywh: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	xyxy = xywh.clone()
	if xyxy.ndim == 1:
		xyxy[2] = xyxy[2] + xyxy[0]
		xyxy[3] = xyxy[3] + xyxy[1]
	elif xyxy.ndim == 2:
		xyxy[:, 2] = xyxy[:, 2] + xyxy[:, 0]
		xyxy[:, 3] = xyxy[:, 3] + xyxy[:, 1]
	else:
		raise ValueError(f"Farray dimensions {xyxy.ndim} is not supported.")
	return xyxy


@dispatch(np.ndarray)
def bbox_xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	xyxy = xywh.copy()
	if xyxy.ndim == 1:
		xyxy[2] = xyxy[2] + xyxy[0]
		xyxy[3] = xyxy[3] + xyxy[1]
	elif xyxy.ndim == 2:
		xyxy[:, 2] = xyxy[:, 2] + xyxy[:, 0]
		xyxy[:, 3] = xyxy[:, 3] + xyxy[:, 1]
	else:
		raise ValueError(f"Farray dimensions {xyxy.ndim} is not supported.")
	return xyxy


@dispatch(Tensor)
def bbox_xyxy_to_cxcyar(xyxy: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xyxy.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		width     = xyxy[2] - xyxy[0]
		height    = xyxy[3] - xyxy[1]
		cxcyrh[0] = cxcyrh[0] + (width / 2.0)
		cxcyrh[1] = cxcyrh[1] + (height / 2.0)
		cxcyrh[2] = (width * height)
		cxcyrh[3] = (width / height)
	elif cxcyrh.ndim == 2:
		widths       = xyxy[:, 2] - xyxy[:, 0]
		heights      = xyxy[:, 3] - xyxy[:, 1]
		cxcyrh[:, 0] = cxcyrh[:, 0] + (widths / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (heights / 2.0)
		cxcyrh[:, 2] = (widths * heights)
		cxcyrh[:, 3] = (widths / heights)
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_xyxy_to_cxcyar(xyxy: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xyxy.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		width     = xyxy[2] - xyxy[0]
		height    = xyxy[3] - xyxy[1]
		cxcyrh[0] = cxcyrh[0] + (width / 2.0)
		cxcyrh[1] = cxcyrh[1] + (height / 2.0)
		cxcyrh[2] = (width * height)
		cxcyrh[3] = (width / height)
	elif cxcyrh.ndim == 2:
		widths       = xyxy[:, 2] - xyxy[:, 0]
		heights      = xyxy[:, 3] - xyxy[:, 1]
		cxcyrh[:, 0] = cxcyrh[:, 0] + (widths / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (heights / 2.0)
		cxcyrh[:, 2] = (widths * heights)
		cxcyrh[:, 3] = (widths / heights)
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(Tensor)
def bbox_xyxy_to_cxcyrh(xyxy: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xyxy.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		width     = xyxy[2] - xyxy[0]
		height    = xyxy[3] - xyxy[1]
		cxcyrh[0] = cxcyrh[0] + (width / 2.0)
		cxcyrh[1] = cxcyrh[1] + (height / 2.0)
		cxcyrh[2] = (width / height)
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = xyxy[:, 2] - xyxy[:, 0]
		heights      = xyxy[:, 3] - xyxy[:, 1]
		cxcyrh[:, 0] = cxcyrh[:, 0] + (widths / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (heights / 2.0)
		cxcyrh[:, 2] = (widths / heights)
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_xyxy_to_cxcyrh(xyxy: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xyxy.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		width     = xyxy[2] - xyxy[0]
		height    = xyxy[3] - xyxy[1]
		cxcyrh[0] = cxcyrh[0] + (width / 2.0)
		cxcyrh[1] = cxcyrh[1] + (height / 2.0)
		cxcyrh[2] = (width / height)
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = xyxy[:, 2] - xyxy[:, 0]
		heights      = xyxy[:, 3] - xyxy[:, 1]
		cxcyrh[:, 0] = cxcyrh[:, 0] + (widths / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (heights / 2.0)
		cxcyrh[:, 2] = (widths / heights)
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(Tensor)
def bbox_xyxy_to_cxcywh(xyxy: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x, center_y, width, height].
	"""
	cxcyrh = xyxy.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		width     = xyxy[2] - xyxy[0]
		height    = xyxy[3] - xyxy[1]
		cxcyrh[0] = cxcyrh[0] + (width / 2.0)
		cxcyrh[1] = cxcyrh[1] + (height / 2.0)
		cxcyrh[2] = width
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = xyxy[:, 2] - xyxy[:, 0]
		heights      = xyxy[:, 3] - xyxy[:, 1]
		cxcyrh[:, 0] = cxcyrh[:, 0] + (widths / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (heights / 2.0)
		cxcyrh[:, 2] = widths
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_xyxy_to_cxcywh(xyxy: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x, center_y, width, height].
	"""
	cxcyrh = xyxy.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		width     = xyxy[2] - xyxy[0]
		height    = xyxy[3] - xyxy[1]
		cxcyrh[0] = cxcyrh[0] + (width / 2.0)
		cxcyrh[1] = cxcyrh[1] + (height / 2.0)
		cxcyrh[2] = width
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = xyxy[:, 2] - xyxy[:, 0]
		heights      = xyxy[:, 3] - xyxy[:, 1]
		cxcyrh[:, 0] = cxcyrh[:, 0] + (widths / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (heights / 2.0)
		cxcyrh[:, 2] = widths
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"Farray dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(Tensor, (int, float), (int, float))
def bbox_xyxy_to_cxcywh_norm(xyxy: Tensor, height, width) -> Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_xyxy_to_cxcywh(xyxy)
	cxcywh_norm = cxcywh_norm.float()
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] /= width
		cxcywh_norm[1] /= height
		cxcywh_norm[2] /= width
		cxcywh_norm[3] /= height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"Farray dimensions {cxcywh_norm.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_xyxy_to_cxcywh_norm(xyxy: np.ndarray, height, width) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_xyxy_to_cxcywh(xyxy)
	cxcywh_norm = cxcywh_norm.astype(float)
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] /= width
		cxcywh_norm[1] /= height
		cxcywh_norm[2] /= width
		cxcywh_norm[3] /= height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"Farray dimensions {cxcywh_norm.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(Tensor)
def bbox_xyxy_to_xywh(xyxy: Tensor) -> Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[top_left_x, top_left_y, width, height].
	"""
	xywh = xyxy.clone()
	xywh = xywh.float()
	if xywh.ndim == 1:
		xywh[2] = xywh[2] - xywh[0]
		xywh[3] = xywh[3] - xywh[1]
	elif xywh.ndim == 2:
		xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
		xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
	else:
		raise ValueError(f"Farray dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(np.ndarray)
def bbox_xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[top_left_x, top_left_y, width, height].
	"""
	xywh = xyxy.copy()
	xywh = xywh.astype(float)
	if xywh.ndim == 1:
		xywh[2] = xywh[2] - xywh[0]
		xywh[3] = xywh[3] - xywh[1]
	elif xywh.ndim == 2:
		xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
		xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
	else:
		raise ValueError(f"Farray dimensions {xywh.ndim} is not supported.")
	return xywh


# MARK: - Bbox <-> Mask

def bbox_to_mask(boxes: Tensor, width: int, height: int) -> Tensor:
	"""Convert 2D bounding boxes to masks. Covered area is 1. and the remaining
	is 0.

	Args:
		boxes (Tensor):
			A tensor containing the coordinates of the bounding boxes to be
			extracted. Image must have the shape of [B, 4, 2], where each
			box is defined in the following `clockwise` order:
			top-left, top-right, bottom-right and bottom-left.
			Fcoordinates must be in the x, y order.
		width (int):
			Width of the masked image.
		height (int):
			Height of the masked image.

	Returns:
		(Tensor):
			Output mask image.

	Note:
		It is currently non-differentiable.

	Examples:
		>>> boxes = Tensor([[
		...        [1., 1.],
		...        [3., 1.],
		...        [3., 2.],
		...        [1., 2.],
		...   ]])  # [1, 4, 2]
		>>> bbox_to_mask(boxes, 5, 5)
		image([[[0., 0., 0., 0., 0.],
				 [0., 1., 1., 1., 0.],
				 [0., 1., 1., 1., 0.],
				 [0., 0., 0., 0., 0.],
				 [0., 0., 0., 0., 0.]]])
	"""
	validate_bbox(boxes)
	# Zero padding the surroundings
	mask = torch.zeros(
		(len(boxes), height + 2, width + 2), dtype=torch.float,
		device=boxes.device
	)
	# Push all points one pixel off in order to zero-out the fully filled rows
	# or columns
	box_i = (boxes + 1).long()
	# Set all pixels within box to 1
	for msk, bx in zip(mask, box_i):
		msk[bx[0, 1]:bx[2, 1] + 1, bx[0, 0]:bx[1, 0] + 1] = 1.0
	return mask[:, 1:-1, 1:-1]


# MARK: - Geometric Properties

def bbox_area(xyxy: np.ndarray) -> float:
	"""Calculate the bbox area with the bounding box of format
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	return (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
	
	
@dispatch(Tensor)
def bbox_xyxy_center(xyxy: Tensor) -> Tensor:
	"""Return the center of the box of format
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	xyah = bbox_xyxy_to_cxcywh(xyxy)
	xyah = xyah.float()
	if xyah.ndim == 1:
		return xyah[0:2]
	elif xyah.ndim == 2:
		return xyah[:, 0:2]
	else:
		raise ValueError(f"Farray dimensions {xyah.ndim} is not supported.")
	

@dispatch(np.ndarray)
def bbox_xyxy_center(xyxy: np.ndarray) -> np.ndarray:
	"""Return the center of the box of format
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	xyah = bbox_xyxy_to_cxcywh(xyxy)
	xyah = xyah.astype(float)
	if xyah.ndim == 1:
		return xyah[0:2]
	elif xyah.ndim == 2:
		return xyah[:, 0:2]
	else:
		raise ValueError(f"Farray dimensions {xyah.ndim} is not supported.")


def infer_bbox_shape(boxes: Tensor) -> ListOrTuple2T[Tensor]:
	"""Auto-infer the output sizes for the given 2D bounding boxes.

	Args:
		boxes (Tensor):
			A tensor containing the coordinates of the bounding boxes to be
			extracted. Image must have the shape of [B, 4, 2], where each
			box is defined in the following `clockwise`
			 order: top-left, top-right, bottom-right, bottom-left.
			 Fcoordinates must be in the x, y order.

	Returns:
		- Bounding box heights, shape of [B].
		- Boundingbox widths, shape of [B].

	Example:
		>>> boxes = Tensor([[
		...     [1., 1.],
		...     [2., 1.],
		...     [2., 2.],
		...     [1., 2.],
		... ], [
		...     [1., 1.],
		...     [3., 1.],
		...     [3., 2.],
		...     [1., 2.],
		... ]])  # 2x4x2
		>>> infer_bbox_shape(boxes)
		(image([2., 2.]), image([2., 3.]))
	"""
	validate_bbox(boxes)
	width  = boxes[:, 1, 0] - boxes[:, 0, 0] + 1
	height = boxes[:, 2, 1] - boxes[:, 0, 1] + 1
	return height, width


# MARK: - Box-Box Interaction

def bbox_ioa(xyxy1: np.ndarray, xyxy2: np.ndarray) -> np.ndarray:
	"""Calculate the intersection over area given xyxy1, xyxy2.
	
	Args:
		xyxy1 (np.ndarray):
			A single bounding box as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
		xyxy2 (np.ndarray):
			An array of bounding boxes as
			[:, top_left_x, top_left_y, bottom_right_x, bottom_right_y].
			
	Returns:
		ioa (np.ndarray):
			Fioa metrics.
	"""
	xyxy2 = xyxy2.transpose()
	
	# Get the coordinates of bounding boxes
	b1_x1, b1_y1, b1_x2, b1_y2 = xyxy1[0], xyxy1[1], xyxy1[2], xyxy1[3]
	b2_x1, b2_y1, b2_x2, b2_y2 = xyxy2[0], xyxy2[1], xyxy2[2], xyxy2[3]
	
	# Intersection area
	inter_area = \
		(np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) \
		* (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
		
	# bbox2 area
	bbox2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16
	
	# Intersection over box2 area
	return inter_area / bbox2_area


@dispatch(Tensor, Tensor)
def bbox_iou(xyxy1: Tensor, xyxy2: Tensor) -> float:
	"""Find the Intersection over Union (IoU) between two 2 boxes.

	Args:
		xyxy1 (Tensor):
			Ftarget bounding box as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
		xyxy2 (Tensor):
			Ground-truth bounding box as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].

	Returns:
		iou (float):
			Fratio IoU.
	"""
	xyxy1_np = xyxy1.numpy()
	xyxy2_np = xyxy2.numpy()
	return bbox_iou(xyxy1_np, xyxy2_np)


@dispatch(np.ndarray, np.ndarray)
def bbox_iou(xyxy1: np.ndarray, xyxy2: np.ndarray) -> float:
	"""Find the Intersection over Union (IoU) between two 2 boxes.

	Args:
		xyxy1 (np.ndarray):
			Ftarget bounding box as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
		xyxy2 (np.ndarray):
			Ground-truth bounding box as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].

	Returns:
		iou (float):
			Fratio IoU.
	"""
	xx1 = np.maximum(xyxy1[0], xyxy2[0])
	yy1 = np.maximum(xyxy1[1], xyxy2[1])
	xx2 = np.minimum(xyxy1[2], xyxy2[2])
	yy2 = np.minimum(xyxy1[3], xyxy2[3])
	w   = np.maximum(0.0, xx2 - xx1)
	h   = np.maximum(0.0, yy2 - yy1)
	wh  = w * h
	ou  = wh / ((xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1]) +
				(xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1]) - wh)
	return ou


@dispatch(np.ndarray, np.ndarray)
def batch_bbox_iou(xyxy1: np.ndarray, xyxy2: np.ndarray) -> np.ndarray:
	"""From SORT: Computes IOU between two sets of boxes.

	Args:
		xyxy1 (np.ndarray):
			Ftarget bounding boxes as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
		xyxy2 (np.ndarray):
			Ground-truth bounding boxes as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].

	Returns:
		iou (np.ndarray):
			Fratio IoUs.
	"""
	xyxy1 = np.expand_dims(xyxy1, 1)
	xyxy2 = np.expand_dims(xyxy2, 0)
	xx1   = np.maximum(xyxy1[..., 0], xyxy2[..., 0])
	yy1   = np.maximum(xyxy1[..., 1], xyxy2[..., 1])
	xx2   = np.minimum(xyxy1[..., 2], xyxy2[..., 2])
	yy2   = np.minimum(xyxy1[..., 3], xyxy2[..., 3])
	w     = np.maximum(0.0, xx2 - xx1)
	h     = np.maximum(0.0, yy2 - yy1)
	wh    = w * h
	iou   = wh / ((xyxy1[..., 2] - xyxy1[..., 0]) *
				  (xyxy1[..., 3] - xyxy1[..., 1]) +
				  (xyxy2[..., 2] - xyxy2[..., 0]) *
				  (xyxy2[..., 3] - xyxy2[..., 1]) - wh)
	return iou


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
	"""Perform non-maxima suppression (NMS) on a given image of bounding boxes
	according to the intersection-over-union (IoU).

	Args:
		boxes (Tensor):
			Tensor containing the encoded bounding boxes with the shape
			[N, [x_1, y_1, x_2, y_2] ].
		scores (Tensor)::
			Tensor containing the scores associated to each bounding box with
			shape [N,].
		iou_threshold (float):
			Fthroshold to discard the overlapping boxes.

	Return:
		(Tensor):
			A image mask with the indices to keep from the input set of boxes and scores.

	Example:
		>>> boxes  = Tensor([
		...     [10., 10., 20., 20.],
		...     [15., 5., 15., 25.],
		...     [100., 100., 200., 200.],
		...     [100., 100., 200., 200.]])
		>>> scores = Tensor([0.9, 0.8, 0.7, 0.9])
		>>> nms(boxes, scores, iou_threshold=0.8)
		image([0, 3, 1])
	"""
	if len(boxes.shape) != 2 and boxes.shape[-1] != 4:
		raise ValueError(f"boxes expected as [N, 4]. Got: {boxes.shape}.")
	if len(scores.shape) != 1:
		raise ValueError(f"scores expected as [N]. Got: {scores.shape}.")
	if boxes.shape[0] != scores.shape[0]:
		raise ValueError(f"boxes and scores mus have same shape. "
						 f"Got: {boxes.shape, scores.shape}.")

	x1, y1, x2, y2 = boxes.unbind(-1)
	areas 	       = (x2 - x1) * (y2 - y1)
	_, order       = scores.sort(descending=True)

	keep = []
	while order.shape[0] > 0:
		i   = order[0]
		keep.append(i)
		xx1 = torch.max(x1[i], x1[order[1:]])
		yy1 = torch.max(y1[i], y1[order[1:]])
		xx2 = torch.min(x2[i], x2[order[1:]])
		yy2 = torch.min(y2[i], y2[order[1:]])

		w     = torch.clamp(xx2 - xx1, min=0.)
		h     = torch.clamp(yy2 - yy1, min=0.)
		inter = w * h
		ovr   = inter / (areas[i] + areas[order[1:]] - inter)

		inds  = torch.where(ovr <= iou_threshold)[0]
		order = order[inds + 1]

	if len(keep) > 0:
		return torch.stack(keep)
	return torch.tensor(keep)


# MARK: - Transformation

def clip_bbox_xyxy(xyxy: Tensor, image_size: Dim2) -> Tensor:
	"""Clip bounding boxes to image size [H, W].

	Args:
		xyxy (Tensor):
			Bounding boxes coordinates as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
		image_size (Dim2):
			Image size as [H, W].

	Returns:
		box_xyxy (Tensor):
			Clipped bounding boxes.
	"""
	xyxy[:, 0].clamp_(0, image_size[1])  # x1
	xyxy[:, 1].clamp_(0, image_size[0])  # y1
	xyxy[:, 2].clamp_(0, image_size[1])  # x2
	xyxy[:, 3].clamp_(0, image_size[0])  # y2
	return xyxy


def cutout_bbox(
	image: np.ndarray, bbox_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
	"""Applies image cutout augmentation with bounding box labels.

	References:
		https://arxiv.org/abs/1708.04552

	Args:
		image (np.ndarray):
			Image.
		bbox_labels (np.ndarray):
			Bounding box labels where the bbox coordinates are located at:
			labels[:, 2:6].

	Returns:
		image_cutout (np.ndarray):
			Cutout image.
		bbox_labels_cutout (np.ndarray):
			Cutout labels.
	"""
	h, w               = image.shape[:2]
	image_cutout       = image.copy()
	bbox_labels_cutout = bbox_labels.copy()
	
	# NOTE: Create random masks
	scales = ([0.5] * 1 +
			  [0.25] * 2 +
			  [0.125] * 4 +
			  [0.0625] * 8 +
			  [0.03125] * 16)  # image size fraction
	for s in scales:
		mask_h = random.randint(1, int(h * s))
		mask_w = random.randint(1, int(w * s))
		
		# Box
		xmin = max(0, random.randint(0, w) - mask_w // 2)
		ymin = max(0, random.randint(0, h) - mask_h // 2)
		xmax = min(w, xmin + mask_w)
		ymax = min(h, ymin + mask_h)
		
		# Apply random color mask
		image_cutout[ymin:ymax, xmin:xmax] = [random.randint(64, 191)
											  for _ in range(3)]
		
		# Return unobscured bounding boxes
		if len(bbox_labels_cutout) and s > 0.03:
			box = np.array([xmin, ymin, xmax, ymax], np.float32)
			# Intersection over area
			ioa = bbox_ioa(box, bbox_labels_cutout[:, 2:6])
			# Remove >60% obscured labels
			bbox_labels_cutout = bbox_labels_cutout[ioa < 0.60]
	
	return image_cutout, bbox_labels_cutout


"""
def bbox_random_perspective(
	image      : Tensor,
	bbox       : Tensor,
	rotate     : float    = 10,
	translate  : float    = 0.1,
	scale      : float    = 0.1,
	shear      : float    = 10,
	perspective: float    = 0.0,
	border     : Sequence = (0, 0)
) -> tuple[Tensor, Tensor]:
	height    = image.shape[0] + border[0] * 2  # Shape of [HWC]
	width     = image.shape[1] + border[1] * 2
	image_new = image.clone()
	bbox_new  = bbox.clone()
	
	# NOTE: Center
	C       = torch.eye(3)
	C[0, 2] = -image_new.shape[2] / 2  # x translation (pixels)
	C[1, 2] = -image_new.shape[1] / 2  # y translation (pixels)
	
	# NOTE: Perspective
	P       = torch.eye(3)
	P[2, 0] = Uniform(-perspective, perspective).sample((1,))
	# x perspective (about y)
	P[2, 1] = Uniform(-perspective, perspective).sample((1,))
	# y perspective (about x)
	
	# NOTE: Rotation and Scale
	R = torch.eye(3)
	a = Uniform(-rotate, rotate).sample((1,))
	# Add 90deg rotations to small rotations
	# a += random.choice([-180, -90, 0, 90])
	s = Uniform(1 - scale, 1 + scale).sample((1,))
	# s = 2 ** random.uniform(-scale, scale)
	R[:2] = get_rotation_matrix2d(center=torch.tensor((0, 0)), angle=a, scale=s)
	
	# NOTE: Shear
	S       = torch.eye(3)
	# x shear (deg)
	S[0, 1] = torch.tan(Uniform(-shear, shear).sample((1,)) * pi / 180)
	# y shear (deg)
	S[1, 0] = torch.tan(Uniform(-shear, shear).sample((1,)) * pi / 180)
	
	# NOTE: Translation
	T       = torch.eye(3)
	# x translation (pixels)
	T[0, 2] = Uniform(0.5 - translate, 0.5 + translate).sample((1,)) * width
	# y translation (pixels)
	T[1, 2] = Uniform(0.5 - translate, 0.5 + translate).sample((1,)) * height
	
	# NOTE: Combined rotation matrix
	M = T @ S @ R @ P @ C  # Order of operations (right to left) is IMPORTANT
	
	# NOTE: Image changed
	if (border[0] != 0) or (border[1] != 0) or (M != torch.eye(3)).any():
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
"""


def random_bbox_perspective(
	image      : np.ndarray,
	bbox       : np.ndarray = (),
	rotate     : float      = 10,
	translate  : float      = 0.1,
	scale      : float      = 0.1,
	shear      : float      = 10,
	perspective: float      = 0.0,
	border     : Dim2       = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
	"""Perform random perspective the image and the corresponding bounding box
	labels.

	Args:
		image (np.ndarray):
			Image of shape [H, W, C].
		bbox (np.ndarray):
			Bounding box labels where the bbox coordinates are located at:
			labels[:, 2:6]. Default: `()`.
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
		border (sequence):

	Returns:
		image_new (np.ndarray):
			Augmented image.
		bbox_new (np.ndarray):
			Augmented bounding boxes.
	"""
	height    = image.shape[0] + border[0] * 2  # Shape of [H, W, C]
	width     = image.shape[1] + border[1] * 2
	image_new = image.copy()
	bbox_new  = bbox.copy()
	
	# NOTE: Center
	C       = np.eye(3)
	C[0, 2] = -image_new.shape[1] / 2  # x translation (pixels)
	C[1, 2] = -image_new.shape[0] / 2  # y translation (pixels)
	
	# NOTE: Perspective
	P       = np.eye(3)
	P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
	P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
	
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

	# NOTE: Transform bboxes' coordinates
	n = len(bbox_new)
	if n:
		# NOTE: Warp points
		xy = np.ones((n * 4, 3))
		xy[:, :2] = bbox_new[:, [2, 3, 4, 5, 2, 5, 4, 3]].reshape(n * 4, 2)
		# x1y1, x2y2, x1y2, x2y1
		xy = xy @ M.T  # Transform
		if perspective:
			xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # Rescale
		else:  # Affine
			xy = xy[:, :2].reshape(n, 8)
		
		# NOTE: Create new boxes
		x  = xy[:, [0, 2, 4, 6]]
		y  = xy[:, [1, 3, 5, 7]]
		xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
		
		# # apply angle-based reduction of bounding boxes
		# radians = a * math.pi / 180
		# reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
		# x = (xy[:, 2] + xy[:, 0]) / 2
		# y = (xy[:, 3] + xy[:, 1]) / 2
		# w = (xy[:, 2] - xy[:, 0]) * reduction
		# h = (xy[:, 3] - xy[:, 1]) * reduction
		# xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
		
		# clip boxes
		xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
		xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
		
		# NOTE: Filter candidates
		i = is_bbox_candidates(
			bbox_new[:, 2:6].T * s, xy.T
		)
		bbox_new = bbox_new[i]
		bbox_new[:, 2:6] = xy[i]
	
	return image_new, bbox_new


def scale_bbox_xyxy(
	xyxy: Tensor, image_size: Dim2, new_size: Dim2, ratio_pad=None
) -> Tensor:
	"""Scale bounding boxes coordinates (from detector size) to the original
	image size.

	Args:
		xyxy (Tensor):
			Bounding boxes coordinates as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
		image_size (Dim2):
			Detector's input size as [H, W].
		new_size (Dim2):
			Original image size as [H, W].
		ratio_pad:

	Returns:
		box_xyxy (Tensor):
			Fscaled bounding boxes.
	"""
	if ratio_pad is None:  # Calculate from new_size
		gain = min(image_size[0] / new_size[0],
				   image_size[1] / new_size[1])  # gain  = old / new
		pad  = (image_size[1] - new_size[1] * gain) / 2, \
			   (image_size[0] - new_size[0] * gain) / 2  # wh padding
	else:
		gain = ratio_pad[0][0]
		pad  = ratio_pad[1]
	
	xyxy[:, [0, 2]] -= pad[0]  # x padding
	xyxy[:, [1, 3]] -= pad[1]  # y padding
	xyxy[:, :4]     /= gain
	return clip_bbox_xyxy(xyxy, new_size)


@dispatch(Tensor, (int, float), (int, float))
def shift_bbox(xyxy: Tensor, ver, hor) -> np.ndarray:
	"""Shift the bounding box with the given `ver` and `hor` values.

	Args:
		xyxy (Tensor):
			Bounding box as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
		ver (int, float):
			Fvertical value to shift.
		hor (int, float):
			Fhorizontal value to shift.

	Returns:
		xyxy_shifted (np.ndarray):
			Fshifted bounding box.
	"""
	xyxy_shift       = xyxy.clone()
	xyxy_shift[:, 0] = xyxy[:, 0] + hor  # pad width
	xyxy_shift[:, 1] = xyxy[:, 1] + ver  # pad height
	xyxy_shift[:, 2] = xyxy[:, 2] + hor
	xyxy_shift[:, 3] = xyxy[:, 3] + ver
	return xyxy_shift


@dispatch(np.ndarray, (int, float), (int, float))
def shift_bbox(xyxy: np.ndarray, ver, hor) -> np.ndarray:
	"""Shift the bounding box with the given `ver` and `hor` values.

	Args:
		xyxy (np.ndarray):
			Bounding box as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
		ver (int, float):
			Fvertical value to shift.
		hor (int, float):
			Fhorizontal value to shift.

	Returns:
		xyxy_shifted (np.ndarray):
			Fshifted bounding box.
	"""
	xyxy_shift       = xyxy.copy()
	xyxy_shift[:, 0] = xyxy[:, 0] + hor  # pad width
	xyxy_shift[:, 1] = xyxy[:, 1] + ver  # pad height
	xyxy_shift[:, 2] = xyxy[:, 2] + hor
	xyxy_shift[:, 3] = xyxy[:, 3] + ver
	return xyxy_shift


# MARK: - Validate

def is_bbox_candidates(
	xyxy1   : np.ndarray,
	xyxy2   : np.ndarray,
	wh_thr  : float = 2,
	ar_thr  : float = 20,
	area_thr: float = 0.2
) -> bool:
	"""Return `True` if xyxy2 is the candidate for xyxy1.
	
	Args:
		xyxy1 (np.ndarray):
			Bounding box before augment as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
		xyxy2 (np.ndarray):
			Bounding box after augment as
			[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
		wh_thr (float):
			Threshold of both width and height (pixels).
		ar_thr (float):
			Aspect ratio threshold.
		area_thr (float):
			Area ratio threshold.
	"""
	w1, h1 = xyxy1[2] - xyxy1[0], xyxy1[3] - xyxy1[1]
	w2, h2 = xyxy2[2] - xyxy2[0], xyxy2[3] - xyxy2[1]
	ar     = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # Aspect ratio
	return ((w2 > wh_thr) &
			(h2 > wh_thr) &
			(w2 * h2 / (w1 * h1 + 1e-16) > area_thr) &
			(ar < ar_thr))  # candidates


@torch.jit.ignore
def validate_bbox(boxes: Tensor) -> bool:
	"""Validate if a 2D bounding box usable or not. This function checks if the
	boxes are rectangular or not.

	Args:
		boxes (Tensor):
			A tensor containing the coordinates of the bounding boxes to be
			extracted. Image must have the shape of [B, 4, 2], where each
			box is defined in the following `clockwise`
			order: top-left, top-right, bottom-right, bottom-left.
			Coordinates must be in the x, y order.
	"""
	if not (len(boxes.shape) == 3 and boxes.shape[1:] == torch.Size([4, 2])):
		raise AssertionError(f"Box shape must be [B, 4, 2]. "
							 f"Got: {boxes.shape}.")
	if not torch.allclose((boxes[:, 1, 0] - boxes[:, 0, 0] + 1),
						  (boxes[:, 2, 0] - boxes[:, 3, 0] + 1)):
		raise ValueError(
			"Boxes must have be rectangular, while get widths %s and %s"
			% (str(boxes[:, 1, 0] - boxes[:, 0, 0] + 1),
			   str(boxes[:, 2, 0] - boxes[:, 3, 0] + 1))
		)
	if not torch.allclose((boxes[:, 2, 1] - boxes[:, 0, 1] + 1),
						  (boxes[:, 3, 1] - boxes[:, 1, 1] + 1)):
		raise ValueError(
			"Boxes must have be rectangular, while get heights %s and %s"
			% (str(boxes[:, 2, 1] - boxes[:, 0, 1] + 1),
			   str(boxes[:, 3, 1] - boxes[:, 1, 1] + 1))
		)
	return True
