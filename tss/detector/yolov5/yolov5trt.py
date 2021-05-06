# ==================================================================== #
# File name: yolov5.py
# Author: Long H. Pham and Duong N.-N. Tran
# Date created: 03/27/2021
#
# ``Yolov5`` class.
# ==================================================================== #
import os
import sys
from collections import OrderedDict
from typing import List, Optional
from typing import Union
import ctypes

import numpy as np

import pycuda.driver as cuda
import tensorrt as trt

import cv2

import torch
import torchvision
import torchvision.transforms.functional as F
from torch import Tensor

import tss.ops as ops
from tss.detector import Detection
from tss.detector import Detector
from tss.utils import is_engine_saved_file
from tss.utils import is_yaml_file
from tss.utils import printe
from .api.models.yolo import Model
from .api.utils.general import non_max_suppression


# MARK: - YOLOv5

class YOLOv5TRT(Detector):
	"""YOLOv5 detector model in TensorRT version.

	Using Engine from pt->wts->engine

	Attributes:
		model_config (str):
			The path to the config file for each yolov5 variant model.
	"""

	# MARK: Magic Functions

	def __init__(self,
				 device: Optional[str] = None,
				 **kwargs):
		super().__init__(**kwargs)
		self.device_index = device
		self.load_model()
		self.should_resize = True

	def __del__(self):
		self.ctx.pop()

	# MARK: Configure

	def load_model(self):
		"""Pipeline to load the model.
		"""
		current_dir = os.path.dirname(os.path.abspath(__file__))  # "...detector/yolov5"

		# TODO: Load plugin
		plugin_library = os.path.join(current_dir, "weights/libmyplugins.so")
		ctypes.CDLL(plugin_library)

		# TODO: Simple check
		# i.e, "yolov5s.pt" means using yolov5s variance.
		if self.weights is None or self.weights == "":
			printe("No weights file has been defined!")
			raise ValueError

		# TODO: Get path to weight file
		self.weights = os.path.join(current_dir, "weights", self.weights)

		# DEBUG:
		print(f"self.weights :: {self.weights}")

		if not is_engine_saved_file(file=self.weights):
			raise FileNotFoundError



		# TODO: Get path to model variant's config
		model_config = os.path.join(current_dir, "configs", f"{self.variant}.yaml")
		if not is_yaml_file(file=model_config):
			raise FileNotFoundError

		# TODO: Define model and load weights
		# Create a Context on this device,
		self.ctx = cuda.Device(int(self.device_index)).make_context()
		stream = cuda.Stream()
		TRT_LOGGER = trt.Logger(trt.Logger.INFO)
		runtime = trt.Runtime(TRT_LOGGER)

		# Deserialize the engine from file
		with open(self.weights, "rb") as f:
			engine = runtime.deserialize_cuda_engine(f.read())
		# TODO: load context to model
		self.model = engine.create_execution_context()

		host_inputs = []
		cuda_inputs = []
		host_outputs = []
		cuda_outputs = []
		bindings = []

		for binding in engine:
			size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
			dtype = trt.nptype(engine.get_binding_dtype(binding))
			# Allocate host and device buffers
			host_mem = cuda.pagelocked_empty(size, dtype)
			cuda_mem = cuda.mem_alloc(host_mem.nbytes)
			# Append the device buffer to device bindings.
			bindings.append(int(cuda_mem))
			# Append to the appropriate list.
			if engine.binding_is_input(binding):
				host_inputs.append(host_mem)
				cuda_inputs.append(cuda_mem)
			else:
				host_outputs.append(host_mem)
				cuda_outputs.append(cuda_mem)

		# Store
		self.stream = stream
		self.engine = engine
		self.host_inputs = host_inputs
		self.cuda_inputs = cuda_inputs
		self.host_outputs = host_outputs
		self.cuda_outputs = cuda_outputs
		self.bindings = bindings

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
		# TODO: Safety check
		if self.model is None:
			printe("Model has not been defined yet!")
			raise NotImplementedError

		# TODO: Forward Pass
		# self.ctx.push()
		batch_detections = self.forward_pass(frame_indexes=frame_indexes, images=images)
		# self.ctx.pop()

		# TODO: Check allowed labels
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
		# self.INPUT_W = 608
		# self.INPUT_H = 608
		self.INPUT_W = self.dims[2]
		self.INPUT_H = self.dims[1]

		inputs = []
		if isinstance(images, np.ndarray):
			for image in images:
				h, w, c = image.shape
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				# Calculate widht and height and paddings
				r_w = self.INPUT_W / w
				r_h = self.INPUT_H / h
				if r_h > r_w:
					tw = self.INPUT_W
					th = int(r_w * h)
					tx1 = tx2 = 0
					ty1 = int((self.INPUT_H - th) / 2)
					ty2 = self.INPUT_H - th - ty1
				else:
					tw = int(r_h * w)
					th = self.INPUT_H
					tx1 = int((self.INPUT_W - tw) / 2)
					tx2 = self.INPUT_W - tw - tx1
					ty1 = ty2 = 0
				# Resize the image with long side while maintaining ratio
				image = cv2.resize(image, (tw, th))
				# Pad the short side with (128,128,128)
				image = cv2.copyMakeBorder(
					image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
				)
				image = image.astype(np.float32)
				# Normalize to [0,1]
				image /= 255.0
				# HWC to CHW format:
				image = np.transpose(image, [2, 0, 1])
				# CHW to NCHW format
				image = np.expand_dims(image, axis=0)
				# Convert the image to row-major order, also known as "C order":
				image = np.ascontiguousarray(image)
				inputs.append(image)
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
		# TODO: Prepare model input
		input_images = self.prepare_input(images=images)

		# TODO: Forward input
		batch_predictions = []
		# Inference
		for input_image in input_images:
			# Make self the active context, pushing it on top of the context stack.
			self.ctx.push()
			# Copy input image to host buffer
			np.copyto(self.host_inputs[0], input_image.ravel())
			# Transfer input data  to the GPU.
			cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
			# Run inference.
			self.model.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
			# Transfer predictions back from the GPU.
			cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
			# Synchronize the stream
			self.stream.synchronize()
			# Remove any context from the top of the context stack, deactivating it.
			self.ctx.pop()

			# Here we use the first row of output in that batch_size = 1
			batch_prediction = self.post_process(
				self.host_outputs[0],
				input_image.shape[2],
				input_image.shape[3],
				images[0].shape[0],
				images[0].shape[1]
			)
			batch_predictions.append(batch_prediction)

		# TODO: Create Detection objects
		batch_detections = []
		for idx, predictions in enumerate(batch_predictions):
			detections = []
			for xyxy, conf, cls in zip(predictions[0], predictions[1], predictions[2]):
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

	def post_process(self, output, detector_h, detector_w, original_h, original_w):
		"""
		description: postprocess the prediction
		param:
			output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
			origin_h:   height of original image
			origin_w:   width of original image
		return:
			result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
			result_scores: finally scores, a tensor, each element is the score correspoing to box
			result_classid: finally classid, a tensor, each element is the classid correspoing to box
		"""
		# Get the num of boxes detected
		num = int(output[0])
		# Reshape to a two dimentional ndarray
		pred = np.reshape(output[1:], (-1, 6))[:num, :]

		# to a torch Tensor
		pred = torch.Tensor(pred).cuda()
		# Get the boxes
		boxes = pred[:, :4]
		# Get the scores
		scores = pred[:, 4]
		# Get the classid
		classid = pred[:, 5]
		# Choose those boxes that score > CONF_THRESH
		si = scores > self.min_confidence
		boxes = boxes[si, :]
		scores = scores[si]
		classid = classid[si]
		# Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
		boxes = self.xywh2xyxy(original_h, original_w, boxes)
		# Do nms
		indices = torchvision.ops.nms(boxes, scores, iou_threshold=self.nms_max_overlap).cpu()
		result_boxes = boxes[indices, :].cpu()
		result_scores = scores[indices].cpu()
		result_classid = classid[indices].cpu()

		# # DEBUG:
		# for idx in enumerate(result_boxes):
		# 	print(result_boxes[idx].item())

		# TODO: Rescale image from model layer (768) to original image size
		# print(detector_h, detector_w)
		# if self.should_resize:
		# 	for boxes in result_boxes:
		# 		boxes[0] = boxes[0] * original_w / detector_w
		# 		boxes[1] = boxes[1] * original_h / detector_h
		# 		boxes[2] = boxes[2] * original_w / detector_w
		# 		boxes[3] = boxes[3] * original_h / detector_h

		return [result_boxes, result_scores, result_classid]

	def xywh2xyxy(self, origin_h, origin_w, x):
		"""
		description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
		param:
			origin_h:   height of original image
			origin_w:   width of original image
			x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
		return:
			y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
		"""
		y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
		r_w = self.INPUT_W / origin_w
		r_h = self.INPUT_H / origin_h
		if r_h > r_w:
			y[:, 0] = x[:, 0] - x[:, 2] / 2
			y[:, 2] = x[:, 0] + x[:, 2] / 2
			y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.INPUT_H - r_w * origin_h) / 2
			y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.INPUT_H - r_w * origin_h) / 2
			y /= r_w
		else:
			y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.INPUT_W - r_h * origin_w) / 2
			y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.INPUT_W - r_h * origin_w) / 2
			y[:, 1] = x[:, 1] - x[:, 3] / 2
			y[:, 3] = x[:, 1] + x[:, 3] / 2
			y /= r_h

		return y


# MARK: - Utils

def _adjust_state_dict(state_dict: OrderedDict):
	od = OrderedDict()
	for key, value in state_dict.items():
		new_key = key.replace("module.", "")
		od[new_key] = value
	return od
