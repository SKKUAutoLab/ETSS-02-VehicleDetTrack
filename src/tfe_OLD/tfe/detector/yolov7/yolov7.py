# ==================================================================== #

# ==================================================================== #
import os
import sys
from collections import OrderedDict
from typing import List
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch import Tensor

import tfe.ops as ops
from tfe.detector import Detection
from tfe.detector import Detector
from tfe.utils import is_torch_saved_file
from tfe.utils import is_yaml_file
from tfe.utils import printe
from .api.models.common import Conv
from .api.models.yolo import Model
from .api.utils.general import non_max_suppression


# MARK: - YOLOv7

class YOLOv7(Detector):
	"""YOLOv5 detector model.
	"""
	
	# MARK: Magic Functions

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.load_model()
		self.model.to(self.device).eval()
		self.should_resize = False

	# MARK: Configure
	
	def load_model(self):
		"""Pipeline to load the model.
		"""
		current_dir = os.path.dirname(os.path.abspath(__file__))  # "...detector/yolov5"

		# NOTE: Simple check
		if self.weights is None or self.weights == "":
			printe("No weights file has been defined!")
			raise ValueError

		# NOTE: Get path to weight file
		self.weights = os.path.join(current_dir, "weights", self.weights)
		if not is_torch_saved_file(file=self.weights):
			raise FileNotFoundError

		# NOTE: Get path to model variant's config
		model_config = os.path.join(current_dir, "configs", f"{self.variant}.yaml")
		if not is_yaml_file(file=model_config):
			raise FileNotFoundError

		# NOTE: Define model and load weights
		# Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
		model = Ensemble()
		ckpt  = torch.load(self.weights)  # load
		model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())

		# NOTE: Compatibility updates
		for m in model.modules():
			if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
				m.inplace = True  # pytorch 1.7.0 compatibility
			elif type(m) is nn.Upsample:
				m.recompute_scale_factor = None  # torch 1.11.0 compatibility
			elif type(m) is Conv:
				m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

		if len(model) == 1:
			self.model = model[-1]
		else:
			print('Ensemble created with %s\n' % self.weights)
			for k in ['names', 'stride']:
				setattr(model, k, getattr(model[-1], k))
			self.model = model  # return ensemble

		# DEBUG

	# MARK: Detection

	def detect_objects(
			self,
			frame_indexes: List[int],
			images: Union[Tensor, np.ndarray]
	) -> Union[List[Detection], List[List[Detection]]]:
		"""Detect road_objects in the image.

		Args:
			frame_indexes (int):
				The list of image indexes in the video.
			images (Tensor or np.array):
				The list of np.array images of shape [BHWC]. If the images is of Tensor type, we assume it has already been normalized.

		Returns:
			batch_detections (list):
				A list of ``Detection``.
				A list of ``Detection`` in batch.
		"""
		# NOTE: Safety check
		if self.model is None:
			printe("Model has not been defined yet!")
			raise NotImplementedError

		# NOTE: Forward Pass
		batch_detections = self.forward_pass(frame_indexes=frame_indexes, images=images)

		# NOTE: Check allowed labels
		[self.suppress_wrong_labels(detections=detections_per_frame) for detections_per_frame in batch_detections]

		return batch_detections

	def prepare_input(self, images: Union[Tensor, np.ndarray]) -> Tensor:
		"""Prepare the model's input for the forward pass.

		Convert to Tensor, resize, change dims to [CHW] and normalize.

		Override this function if you want a custom preparation pipeline.

		Args:
			images (Tensor or np.array):
				The list of np.array images of shape [BHWC]. If the images is of Tensor type, we assume it has already been normalized.

		Returns:
			inputs (Tensor):
				The prepared images [BCHW] with B=1.
		"""
		inputs = None
		if isinstance(images, np.ndarray):
			# if images.shape[2] != self.dims[2]:
			# 	images             = ops.padded_resize_image(images=images, size=self.dims[1:3])
			# 	self.should_resize = True
			inputs = [F.to_tensor(pic=image) for image in images]
			inputs = torch.stack(inputs)
		if torch.is_tensor(inputs) and len(inputs.size()) == 3:
			inputs = inputs.unsqueeze(0)
		return inputs

	def forward_pass(
			self,
			frame_indexes: List[int],
			images: Union[Tensor, np.ndarray]
	) -> Union[List[Detection], List[List[Detection]]]:
		"""Define the forward pass logic of the ``model``.

		Args:
			frame_indexes (int):
				The list of image indexes in the video.
			images (Tensor or np.array):
				The list of np.array images of shape [BHWC]. If the images is of Tensor type, we assume it has already been normalized.

		Returns:
			batch_detections (list):
				A list of ``Detection``.
				A list of ``Detection`` in batch.
		"""
		# NOTE: Prepare model input
		inputs = self.prepare_input(images=images).to(self.device)

		# NOTE: Forward input
		batch_predictions = self.model(inputs, augment=False)[0]
		batch_predictions = non_max_suppression(batch_predictions, self.min_confidence, self.nms_max_overlap,
												classes=self.label_ids)

		# NOTE: Rescale image from model layer (768) to original image size
		if self.should_resize:
			for predictions in batch_predictions:
				det_bbox_xyxy = predictions[:, :4].cpu().detach()
				predictions[:, :4] = ops.scale_bbox_xyxy(
					detector_size=inputs.shape[2:],
					bbox_xyxy=det_bbox_xyxy,
					original_size=images.shape[1:3]
				).round()

		# NOTE: Create Detection objects
		batch_detections = []
		for idx, predictions in enumerate(batch_predictions):
			detections = []
			for *xyxy, conf, cls in predictions:
				bbox_xyxy = np.array([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()], np.int32)
				confident = float(conf)
				class_id = int(cls)
				label = ops.get_label(labels=self.labels, train_id=class_id)
				detections.append(
					Detection(
						frame_index=frame_indexes[0] + idx,
						bbox=bbox_xyxy,
						confidence=confident,
						label=label
					)
				)
			batch_detections.append(detections)

		return batch_detections


# MARK: - Esemble class

class Ensemble(nn.ModuleList):

	'''Ensemble of models'''

	def __init__(self):
		super(Ensemble, self).__init__()

	def forward(self, x, augment=False):
		y = []
		for module in self:
			y.append(module(x, augment)[0])
		# y = torch.stack(y).max(0)[0]  # max ensemble
		# y = torch.stack(y).mean(0)  # mean ensemble
		y = torch.cat(y, 1)  # nms ensemble
		return y, None  # inference, train output


# MARK: - Utils

def _adjust_state_dict(state_dict: OrderedDict):
	od = OrderedDict()
	for key, value in state_dict.items():
		new_key     = key.replace("module.", "")
		od[new_key] = value
	return od
