# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from multipledispatch import dispatch
from torch import Tensor

from core.utils import to_channel_last

__all__ = [
	"_draw_bbox",
	"_draw_pixel",
	"draw_bbox",
	"draw_line",
	"draw_rectangle",
]


# MARK: - Draw Pixel


def _draw_pixel(image: Tensor, x: int, y: int, color: Tensor):
	"""Draws a pixel into an image.

	Args:
		image (Tensor):
			Input image to where to draw the lines with shape [C, H, W].
		x (int):
			Fx coordinate of the pixel.
		y (int):
			Fy coordinate of the pixel.
		color (Tensor):
			Color of the pixel with [C] where `C` is the number of channels
			of the image.
	"""
	image[:, y, x] = color


# MARK: - Draw Line

def draw_line(image: Tensor, p1: Tensor, p2: Tensor, color: Tensor) -> Tensor:
	"""Draw a single line into an image.

	Args:
		image (Tensor):
			Input image to where to draw the lines with shape [C, H, W].
		p1 (Tensor):
			Fstart point [x y] of the line with shape (2).
		p2 (Tensor):
			Fend point [x y] of the line with shape (2).
		color (Tensor):
			Color of the line with shape [C] where `C` is the number of
			channels of the image.

	Return:
		image (Tensor):
			Image with containing the line.

	Examples:
	>>> image = torch.zeros(1, 8, 8)
	>>> draw_line(image, torch.tensor([6, 4]), torch.tensor([1, 4]), torch.tensor([255]))
	image([[[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
			 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
			 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
			 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
			 [  0., 255., 255., 255., 255., 255., 255.,   0.],
			 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
			 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
			 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]]])
	"""
	if (len(p1) != 2) or (len(p2) != 2):
		raise ValueError("p1 and p2 must have length 2.")
	if len(image.size()) != 3:
		raise ValueError("image must have 3 dimensions (C,H,W).")
	if color.size(0) != image.size(0):
		raise ValueError("Color must have the same number of channels as the "
						 "image.")
	if ((p1[0] >= image.size(2)) or
			(p1[1] >= image.size(1) or
			 (p1[0] < 0) or
			 (p1[1] < 0))):
		raise ValueError("p1 is out of bounds.")
	if ((p2[0] >= image.size(2)) or
			(p2[1] >= image.size(1) or
			 (p2[0] < 0) or
			 (p2[1] < 0))):
		raise ValueError("p2 is out of bounds.")

	# Make move arguments to same device and dtype as the input image
	p1, p2, color = p1.to(image), p2.to(image), color.to(image)

	# Assign points
	x1, y1 = p1
	x2, y2 = p2

	# NOTE: Calculate coefficients A,B,C of line from equation Ax + By + C = 0
	A = y2 - y1
	B = x1 - x2
	C = x2 * y1 - x1 * y2

	# Make sure A is positive to utilize the function properly
	if A < 0:
		A = -A
		B = -B
		C = -C

	# NOTE: Calculate the slope of the line
	# Check for division by zero
	if B != 0:
		m = -A / B

	# Make sure you start drawing in the right direction
	x1, x2 = min(x1, x2).long(), max(x1, x2).long()
	y1, y2 = min(y1, y2).long(), max(y1, y2).long()

	# Line equation that determines the distance away from the line
	def line_equation(x, y):
		return A * x + B * y + C

	# Vertical line
	if B == 0:
		image[:, y1:y2 + 1, x1] = color
	# Horizontal line
	elif A == 0:
		image[:, y1, x1:x2 + 1] = color
	# Slope between 0 and 1
	elif 0 < m < 1:
		for i in range(x1, x2 + 1):
			_draw_pixel(image, i, y1, color)
			if line_equation(i + 1, y1 + 0.5) > 0:
				y1 += 1
	# Slope >= 1
	elif m >= 1:
		for j in range(y1, y2 + 1):
			_draw_pixel(image, x1, j, color)
			if line_equation(x1 + 0.5, j + 1) < 0:
				x1 += 1
	# Slope < -1
	elif m <= -1:
		for j in range(y1, y2 + 1):
			_draw_pixel(image, x2, j, color)
			if line_equation(x2 - 0.5, j + 1) > 0:
				x2 -= 1
	# Slope between -1 and 0
	elif -1 < m < 0:
		for i in range(x1, x2 + 1):
			_draw_pixel(image, i, y2, color)
			if line_equation(i + 1, y2 - 0.5) > 0:
				y2 -= 1

	return image


# MARK: - Draw Rectangle

def draw_rectangle(
		image    : Tensor,
		rectangle: Tensor,
		color    : Optional[Tensor] = None,
		fill     : Optional[bool]   = None,
) -> Tensor:
	"""Draw N rectangles on a batch of image tensors.

	Args:
		image (Tensor):
			Tensor of shape [B, C, H, W].
		rectangle (Tensor):
			Represents number of rectangles to draw in [B, N, 4].
			N is the number of boxes to draw per batch index [x1, y1, x2, y2]
			4 is in (top_left.x, top_left.y, bot_right.x, bot_right.y).
		color (Tensor, optional):
			A size 1, size 3, [B, N, 1], or [B, N, 3] image.
			If `C` is 3, and color is 1 channel it will be broadcasted.
			Default: `None`.
		fill (bool, optional):
			A flag used to fill the boxes with color if `True`. Default: `None`.

	Returns:
		image (Tensor):
			This operation modifies image inplace but also returns the drawn
			image for convenience with same shape the of the input
			[B, C, H, W].

	Example:
		>>> img  = torch.rand(2, 3, 10, 12)
		>>> rect = torch.tensor([[[0, 0, 4, 4]], [[4, 4, 10, 10]]])
		>>> out  = draw_rectangle(img, rect)
	"""
	batch, c, h, w = image.shape
	batch_rect, num_rectangle, num_points = rectangle.shape
	if batch != batch_rect:
		raise AssertionError("Image batch and rectangle batch must be equal")
	if num_points != 4:
		raise AssertionError("Number of points in rectangle must be 4")

	# Clone rectangle, in case it's been expanded assignment from clipping
	# causes problems
	rectangle = rectangle.long().clone()

	# Clip rectangle to hxw bounds
	rectangle[:, :, 1::2] = torch.clamp(rectangle[:, :, 1::2], 0, h - 1)
	rectangle[:, :, ::2]  = torch.clamp(rectangle[:, :, ::2], 0, w - 1)

	if color is None:
		color = torch.tensor([0.0] * c).expand(batch, num_rectangle, c)

	if fill is None:
		fill = False

	if len(color.shape) == 1:
		color = color.expand(batch, num_rectangle, c)

	b, n, color_channels = color.shape

	if color_channels == 1 and c == 3:
		color = color.expand(batch, num_rectangle, c)

	for b in range(batch):
		for n in range(num_rectangle):
			if fill:
				image[
				b, :,
				int(rectangle[b, n, 1]): int(rectangle[b, n, 3] + 1),
				int(rectangle[b, n, 0]): int(rectangle[b, n, 2] + 1),
				] = color[b, n, :, None, None]
			else:
				image[
				b, :,
				int(rectangle[b, n, 1]): int(rectangle[b, n, 3] + 1),
				rectangle[b, n, 0]
				] = color[b, n, :, None]
				image[
				b, :,
				int(rectangle[b, n, 1]): int(rectangle[b, n, 3] + 1),
				rectangle[b, n, 2]
				] = color[b, n, :, None]
				image[
				b, :,
				rectangle[b, n, 1],
				int(rectangle[b, n, 0]): int(rectangle[b, n, 2] + 1)
				] = color[b, n, :, None]
				image[
				b, :,
				rectangle[b, n, 3],
				int(rectangle[b, n, 0]): int(rectangle[b, n, 2] + 1)
				] = color[b, n, :, None]

	return image


# MARK: - Draw BBox

def _draw_bbox(
		image    : np.ndarray,
		labels   : np.ndarray,
		colors   : Optional[list] = None,
		thickness: int            = 5
) -> np.ndarray:
	"""Draw bounding box(es) on image. If given the `colors`, use the color
	index corresponding with the `class_id` of the labels.

	Args:
		image (np.ndarray):
			Can be a 4D batch of numpy array or a single image.
		labels (np.ndarray):
			Bounding box labels where the bounding boxes coordinates are
			located at: labels[:, 2:6]. Also, the bounding boxes are in [
			xyxy] format.
		colors (list, optional):
			List of colors.
		thickness (int):
			Thickness of the bounding box border.

	Returns:
		image (np.ndarray):
			Image with drawn bounding boxes.
	"""
	# NOTE: Draw bbox
	image = np.ascontiguousarray(image, dtype=np.uint8)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	for i, l in enumerate(labels):
		class_id    = int(l[1])
		start_point = l[2:4].astype(np.int)
		end_point   = l[4:6].astype(np.int)
		color		= (255, 255, 255)
		if isinstance(colors, (tuple, list)) and len(colors) >= class_id:
			color = colors[class_id]
		image = cv2.rectangle(
			img=image, pt1=tuple(start_point), pt2=tuple(end_point),
			color=tuple(color), thickness=thickness
		)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image


@dispatch(Tensor, Tensor, list, int)
def draw_bbox(
		image    : Tensor,
		labels   : Tensor,
		colors   : Optional[list] = None,
		thickness: int            = 1
):
	"""Draw bounding box(es) on image(s). If given the `colors`, use the color
	index corresponding with the `class_id` of the labels.

	Args:
		image (Tensor):
			Can be a 4D batch of Tensor or a single image.
		labels (Tensor):
			Bounding box labels where the bounding boxes coordinates are
			located at: labels[:, 2:6]. Also, the bounding boxes are in
			[xyxy] format.
		colors (list, optional):
			List of colors.
		thickness (int):
			Thickness of the bounding box border.
	"""
	image_np  = image.numpy()
	labels_np = labels.numpy()
	return draw_bbox(image_np, labels_np, colors, thickness)


@dispatch(np.ndarray, np.ndarray, list, int)
def draw_bbox(
		image    : np.ndarray,
		labels   : np.ndarray,
		colors   : Optional[list] = None,
		thickness: int            = 1
):
	"""Draw bounding box(es) on image(s). If given the `colors`, use the color
	index corresponding with the `class_id` of the labels.

	Args:
		image (np.ndarray):
			Can be a 4D batch of numpy array or a single image.
		labels (np.ndarray):
			Bounding box labels where the bounding boxes coordinates are
			located at: labels[:, 2:6]. Also, the bounding boxes are in
			[xyxy] format.
		colors (list, optional):
			List of colors.
		thickness (int):
			Thickness of the bounding box border.
	"""
	# NOTE: Convert to channel-last
	image = to_channel_last(image)

	# NOTE: Unnormalize image
	from core.enhance import denormalize_naive
	image = denormalize_naive(image)
	image = image.astype(np.uint8)

	# NOTE: If the images are of shape [CHW]
	if image.ndim == 3:
		return _draw_bbox(image, labels, colors, thickness)

	# NOTE: If the images are of shape [BCHW]
	if image.ndim == 4:
		imgs = []
		for i, img in enumerate(image):
			l = labels[labels[:, 0] == i]
			imgs.append(_draw_bbox(img, l, colors, thickness))
		imgs = np.stack(imgs, axis=0).astype(np.unit8)
		return imgs

	raise ValueError(f"Do not support image with ndim: {image.ndim}.")


