# ==================================================================== #
# File name: bbox.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/27/2021
#
# Implements common operations on bbox.
# ==================================================================== #
from typing import Tuple

import numpy as np
# MARK: - Non-modifying Ops
from torch import Tensor


def iou(
	bb_test: np.ndarray,
	bb_gt  : np.ndarray
):
	""" Find the Intersection over Union (IoU) between two 2 bounding box
	
	Args:
		bb_test (np.ndarray):
			The target bounding box
		bb_gt (np.ndarray):
			The groundtruth bounding box
	Returns:
		(float):
			The ratio IoU
	"""
	xx1 = np.maximum(bb_test[0], bb_gt[0])
	yy1 = np.maximum(bb_test[1], bb_gt[1])
	xx2 = np.minimum(bb_test[2], bb_gt[2])
	yy2 = np.minimum(bb_test[3], bb_gt[3])
	w   = np.maximum(0., xx2 - xx1)
	h   = np.maximum(0., yy2 - yy1)
	wh  = w * h
	o   = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
	return (o)


def iou_batch(
	bb_test: np.ndarray,
	bb_gt  : np.ndarray
):
	"""
	From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
	"""
	bb_gt = np.expand_dims(bb_gt, 0)
	bb_test = np.expand_dims(bb_test, 1)
	
	xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
	yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
	xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
	yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
	w = np.maximum(0., xx2 - xx1)
	h = np.maximum(0., yy2 - yy1)
	wh = w * h
	o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
	          + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
	return (o)


def bbox_xywh(bbox_xyxy: np.ndarray) -> np.ndarray:
	"""Return bbox format from [top_left x, top_left y, bottom_right x, bottom_right y]
	to [top_left x, top_left y, width, height].
	
	Args:
		bbox_xyxy (np.ndarray):
		
	Returns:
		bbox (np.ndarray):
			The new bbox_xyah.
	"""
	bbox    = bbox_xyxy.copy()
	width   = bbox[3] - bbox[1]
	height  = bbox[2] - bbox[0]
	bbox[2] = width
	bbox[3] = height
	return bbox


def bbox_xyah(bbox_xyxy: np.ndarray) -> np.ndarray:
	"""Return bbox format from [top_left x, top_left y, bottom_right x, bottom_right y]
	to [center x, center y, aspect ratio, height], where the aspect ratio is ``width / height``.
	
	Args:
		bbox_xyxy (np.ndarray):
		
	Returns:
		bbox (np.ndarray):
			The new bbox_xyah
	"""
	bbox    = bbox_xyxy.copy()
	width   = bbox[3] - bbox[1]
	height  = bbox[2] - bbox[0]
	bbox[0] = bbox[0] + height / 2
	bbox[1] = bbox[1] + width / 2
	bbox[2] = width / height
	bbox[3] = height
	return bbox


def bbox_xyxy_center(bbox_xyxy: np.ndarray) -> np.ndarray:
	"""Return the center of the bbox of format [top_left x, top_left y, bottom_right x, bottom_right y].
	
	Args:
		bbox_xyxy (np.ndarray):
	
	Returns:
		(x, y) (np.ndarray):
			The center point.
	"""
	bbox = bbox_xyah(bbox_xyxy=bbox_xyxy)
	return bbox[0:2]


def bbox_xyxy_to_z(bbox_xyxy: np.ndarray) -> np.ndarray:
	"""Converting bounding box for Kalman Filter.
	
	Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r]
	
	Args:
		bbox_xyxy (np.ndarray):
			The bounding box with form [x1,y1,x2,y2]
	Returns:
		z (np.ndarray):
			The bounding box with form [x,y,s,r]
				where x,y is the centre of the box
				s is the scale/area
				r is the aspect ratio
	"""
	w = bbox_xyxy[2] - bbox_xyxy[0]
	h = bbox_xyxy[3] - bbox_xyxy[1]
	x = bbox_xyxy[0] + w / 2.
	y = bbox_xyxy[1] + h / 2.
	s = w * h
	r = w / float(h)
	return np.array([x, y, s, r]).reshape((4, 1))


def x_to_bbox_xyxy(x: np.ndarray, score: float = None) -> np.ndarray:
	"""Return bounding box from Kalman Filter.
	
	Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
		[x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right

	Args:
		x (np.ndarray):
			The bounding box with form [x,y,s,r]
				where x,y is the centre of the box
				s is the scale/area
				r is the aspect ratio
		score (float):
			The confident score
	Returns:
		bbox (np.ndarray):
			The bounding box with form [x1,y1,x2,y2]
				where x1,y1 is the top left
				x2,y2 is the bottom right
	"""
	w = np.sqrt(x[2] * x[3])
	h = x[2] / w
	if score is None:
		return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
	else:
		return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))
	

# MARK: - Modifying Ops

def scale_bbox_xyxy(
	detector_size : Tuple[int, int],
	bbox_xyxy     : Tensor,
	original_size : Tuple[int, int],
	ratio_pad     = None
) -> Tensor:
	"""Scale bbox coordinates (from detector size) to the original image size.

	Args:
		detector_size (tuple):
			The detector's input size as [H, W].
		bbox_xyxy (Tensor):
			The bbox coordinates as [x, y, x, y].
		original_size (tuple):
			The original image size as [H, W].
		ratio_pad:

	Returns:
		bbox_xyxy (Tensor):
			The scaled bbox.
	"""
	if ratio_pad is None:  # calculate from original_size
		gain = min(detector_size[0] / original_size[0], detector_size[1] / original_size[1])  # gain  = old / new
		pad  = (detector_size[1] - original_size[1] * gain) / 2, (detector_size[0] - original_size[0] * gain) / 2  # wh padding
	else:
		gain = ratio_pad[0][0]
		pad  = ratio_pad[1]

	bbox_xyxy[:, [0, 2]] -= pad[0]  # x padding
	bbox_xyxy[:, [1, 3]] -= pad[1]  # y padding
	bbox_xyxy[:, :4]     /= gain
	return clip_bbox_xyxy(bbox_xyxy=bbox_xyxy, image_size=original_size)


def clip_bbox_xyxy(
	bbox_xyxy : Tensor,
	image_size: Tuple[int, int]
) -> Tensor:
	"""Clip bounding xyxy bounding boxes to image size [H, W].
	
	Args:
		bbox_xyxy (Tensor):
			The bbox coordinates as [x, y, x, y].
		image_size (tuple):
			The image size as [H, W].

	Returns:
		bbox_xyxy (Tensor):
			The clipped bbox.
	"""
	#
	bbox_xyxy[:, 0].clamp_(0, image_size[1])  # x1
	bbox_xyxy[:, 1].clamp_(0, image_size[0])  # y1
	bbox_xyxy[:, 2].clamp_(0, image_size[1])  # x2
	bbox_xyxy[:, 3].clamp_(0, image_size[0])  # y2
	return bbox_xyxy
