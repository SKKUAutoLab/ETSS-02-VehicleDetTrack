#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YOLOv5 object_detectors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
from torch import Tensor

from core.io.filedir import is_torch_saved_file
from core.utils.bbox import scale_bbox_xyxy
from core.utils.geometric_transformation import padded_resize
from core.utils.image import to_tensor
from core.utils.image import is_channel_first
from core.factory.builder import DETECTORS
from core.objects.instance import Instance
from detectors.base import BaseDetector

# NOTE: add model YOLOv5 source to here
sys.path.append('src/detectors/ultralytics')

from detectors.ultralytics.ultralytics import YOLO

from ultralytics import yolo  # noqa
from ultralytics.nn.tasks import (
	ClassificationModel,
	DetectionModel,
	SegmentationModel,
	attempt_load_one_weight,
	attempt_load_weights,
	guess_model_task, nn
)
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, ONLINE, RANK, ROOT,
									callbacks, is_git_dir, is_pip_package, yaml_load)
from ultralytics.yolo.utils.checks import check_imgsz, check_pip_update, check_yaml
from ultralytics.yolo.utils.torch_utils import smart_inference_mode

__all__ = [
	"YOLOv8"
]


# MARK: - YOLOv8

@DETECTORS.register(name="yolov8_trafficsafety")
class YOLOv8(BaseDetector):
	"""YOLOv8 object detector."""

	# MARK: Magic Functions

	def __init__(self,
				 name: str = "yolov8_trafficsafety",
				 *args, **kwargs):
		super().__init__(name=name, *args, **kwargs)

	# MARK: Configure

	def init_model(self):
		"""Create and load model from weights."""
		# NOTE: Create model
		path = self.weights
		# if not is_torch_saved_file(path):
		# 	path, _ = os.path.splitext(path)
		# 	path    = os.path.join(models_zoo_dir, f"{path}.pt")
		# assert is_torch_saved_file(path), f"Not a weights file: {path}"

		# NOTE: load model
		# self.model = YOLO(path)
		# DEBUG:
		self.model = YOLOEnsemble()
		self.model._load(path)
		# self.model.to(device=self.device)

		# Get image size of detector
		if is_channel_first(np.ndarray(self.shape)):
			self.img_size = self.shape[2]
		else:
			self.img_size = self.shape[0]

		# DEBUG:
		# print("*************")
		# print(dir(self.model))
		# print(self.model.overrides)
		# print("*************")

	# MARK: Detection

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
		# NOTE: Safety check
		if self.model is None:
			print("Model has not been defined yet!")
			raise NotImplementedError

		# NOTE: Preprocess
		input = self.preprocess(images=images)
		# NOTE: Forward
		pred  = self.forward(input)
		# NOTE: Postprocess
		instances = self.postprocess(
			indexes=indexes, images=images, input=input, pred=pred
		)
		# NOTE: Suppression
		instances = self.suppress_wrong_labels(instances=instances)

		return instances

	def preprocess(self, images: np.ndarray) -> Tensor:
		"""Preprocess the input images to model's input image.

		Args:
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			input (Tensor):
				Models' input.
		"""
		input = images
		# if self.shape:
		# 	input = padded_resize(input, self.shape, stride=self.stride)
		# 	self.resize_original = True
		# #input = [F.to_tensor(i) for i in input]
		# #input = torch.stack(input)
		# input = to_tensor(input, normalize=True)
		# input = input.to(self.device)
		return input

	def forward(self, input: Tensor) -> Tensor:
		"""Forward pass.

		Args:
			input (Tensor):
				Input image of shape [B, C, H, W].

		Returns:
			pred (Tensor):
				Predictions.
		"""
		# DEBUG:
		# print(dir(self.model))
		# print(self.model.overrides)
		# sys.exit()

		pred = self.model(
			input,
			imgsz   = self.img_size,
			conf    = self.min_confidence,
			iou     = self.nms_max_overlap,
			classes = self.allowed_ids,
			augment = True,
			verbose = False,
		)
		return pred

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
				List of `Instances` objects.
		"""
		# NOTE: Create Detection objects
		instances = []
		# DEBUG:
		# print("******")
		# for result in pred:
		# 	# detection
		# 	result.boxes.xyxy  # box with xyxy format, (N, 4)
		# 	result.boxes.xywh  # box with xywh format, (N, 4)
		# 	result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
		# 	result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
		# 	result.boxes.conf  # confidence score, (N, 1)
		# 	result.boxes.cls  # cls, (N, 1)
		# print("******")

		for idx, result in enumerate(pred):
			inst = []
			xyxys = result.boxes.xyxy.cpu().numpy()
			confs = result.boxes.conf.cpu().numpy()
			clses = result.boxes.cls.cpu().numpy()

			# DEBUG:
			# print(len(confs))

			for bbox_xyxy, conf, cls in zip(xyxys, confs, clses):
				confident   = float(conf)
				class_id    = int(cls)
				class_label = self.class_labels.get_class_label(
					key="train_id", value=class_id
				)
				inst.append(
					Instance(
						frame_index = indexes[0] + idx,
						bbox        = bbox_xyxy,
						confidence  = confident,
						class_label = class_label
					)
				)
			instances.append(inst)
		return instances


# MARK: - Ensemble class of YOLOv8 - Custom

# Map head to model, trainer, validator, and predictor classes

TASK_MAP = {
	'classify': [
		ClassificationModel, yolo.v8.classify.ClassificationTrainer, yolo.v8.classify.ClassificationValidator,
		yolo.v8.classify.ClassificationPredictor],
	'detect': [
		DetectionModel, yolo.v8.detect.DetectionTrainer, yolo.v8.detect.DetectionValidator,
		yolo.v8.detect.DetectionPredictor],
	'segment': [
		SegmentationModel, yolo.v8.segment.SegmentationTrainer, yolo.v8.segment.SegmentationValidator,
		yolo.v8.segment.SegmentationPredictor]}


class YOLOEnsemble:
	"""
	Ensemble YOLO (You Only Look Once) object detection model.

	Args:
		model (str, Path): Path to the model file to load or create.

	Attributes:
		predictor (Any): The predictor object.
		model (Any): The model object.
		trainer (Any): The trainer object.
		task (str): The type of model task.
		ckpt (Any): The checkpoint object if the model loaded from *.pt file.
		cfg (str): The model configuration if loaded from *.yaml file.
		ckpt_path (str): The checkpoint file path.
		overrides (dict): Overrides for the trainer object.
		metrics (Any): The data for metrics.

	Methods:
		__call__(source=None, stream=False, **kwargs):
			Alias for the predict method.
		_new(cfg:str, verbose:bool=True) -> None:
			Initializes a new model and infers the task type from the model definitions.
		_load(weights:str, task:str='') -> None:
			Initializes a new model and infers the task type from the model head.
		_check_is_pytorch_model() -> None:
			Raises TypeError if the model is not a PyTorch model.
		reset() -> None:
			Resets the model modules.
		info(verbose:bool=False) -> None:
			Logs the model info.
		fuse() -> None:
			Fuses the model for faster inference.
		predict(source=None, stream=False, **kwargs) -> List[ultralytics.yolo.engine.results.Results]:
			Performs prediction using the YOLO model.

	Returns:
		list(ultralytics.yolo.engine.results.Results): The prediction results.
	"""

	def __init__(self, model='yolov8n.pt', task=None, session=None) -> None:
		"""
		Initializes the YOLO model.

		Args:
			model (str, Path): model to load or create
		"""
		self._reset_callbacks()
		self.predictor = None  # reuse predictor
		self.model     = None  # model object
		self.trainer   = None  # trainer object
		self.task      = None  # task type
		self.ckpt      = None  # if loaded from *.pt
		self.cfg       = None  # if loaded from *.yaml
		self.ckpt_path = None
		self.overrides = {}  # overrides for trainer object
		self.metrics   = None  # validation/training metrics
		self.session   = session  # HUB session

	def __call__(self, source=None, stream=False, **kwargs):
		return self.predict(source, stream, **kwargs)

	def __getattr__(self, attr):
		name = self.__class__.__name__
		raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

	def _new(self, cfg: str, task=None, verbose=True):
		"""
		Initializes a new model and infers the task type from the model definitions.

		Args:
			cfg (str): model configuration file
			task (str) or (None): model task
			verbose (bool): display model info on load
		"""
		self.cfg   = check_yaml(cfg)  # check YAML
		cfg_dict   = yaml_load(self.cfg, append_filename=True)  # model dict
		# 3 task 'segment' 'classify' 'detect'
		# self.task = 'detect'
		self.task  = task or guess_model_task(cfg_dict)
		self.model = TASK_MAP[self.task][0](cfg_dict, verbose=verbose and RANK == -1)  # build model
		self.overrides['model'] = self.cfg

		# Below added to allow export from yamls
		args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine model and default args, preferring model args
		self.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
		self.model.task = self.task

	def _load(self, weights: str, task=None):
		"""
		Initializes a new model and infers the task type from the model head.

		Args:
			weights (str): model checkpoint to be loaded
			task (str) or (None): model task
		"""
		if isinstance(weights, list):
			# self.model, self.ckpt = attempt_load_weights(weights)
			# self.task = self.model.args['task']
			# self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
			# self.ckpt_path = self.model.pt_path
			self.model, self.ckpt = weights, None
			self.task = task or guess_model_task(weights)
			self.ckpt_path = weights
		elif isinstance(weights, str):
			self.model, self.ckpt = attempt_load_one_weight(weights)
			self.task = self.model.args['task']
			self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
			self.ckpt_path = self.model.pt_path
		self.overrides['model'] = weights

	def _check_is_pytorch_model(self):
		"""
		Raises TypeError is model is not a PyTorch model
		"""
		if not isinstance(self.model, nn.Module):
			raise TypeError(f"model='{self.model}' must be a *.pt PyTorch model, but is a different type. "
							f'PyTorch models can be used to train, val, predict and export, i.e. '
							f"'yolo export model=yolov8n.pt', but exported formats like ONNX, TensorRT etc. only "
							f"support 'predict' and 'val' modes, i.e. 'yolo predict model=yolov8n.onnx'.")

	def _check_pip_update(self):
		"""
		Inform user of ultralytics package update availability
		"""
		if ONLINE and is_pip_package():
			check_pip_update()

	def reset(self):
		"""
		Resets the model modules.
		"""
		self._check_is_pytorch_model()
		for m in self.model.modules():
			if hasattr(m, 'reset_parameters'):
				m.reset_parameters()
		for p in self.model.parameters():
			p.requires_grad = True

	def info(self, verbose=False):
		"""
		Logs model info.

		Args:
			verbose (bool): Controls verbosity.
		"""
		self._check_is_pytorch_model()
		self.model.info(verbose=verbose)

	def fuse(self):
		self._check_is_pytorch_model()
		self.model.fuse()

	@smart_inference_mode()
	def predict(self, source=None, stream=False, **kwargs):
		"""
		Perform prediction using the YOLO model.

		Args:
			source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
						  Accepts all source types accepted by the YOLO model.
			stream (bool): Whether to stream the predictions or not. Defaults to False.
			**kwargs : Additional keyword arguments passed to the predictor.
					   Check the 'configuration' section in the documentation for all available options.

		Returns:
			(List[ultralytics.yolo.engine.results.Results]): The prediction results.
		"""
		if source is None:
			source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
			LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
		is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')) and \
				 ('predict' in sys.argv or 'mode=predict' in sys.argv)

		overrides = self.overrides.copy()
		overrides['conf'] = 0.25
		overrides.update(kwargs)  # prefer kwargs
		overrides['mode'] = kwargs.get('mode', 'predict')
		assert overrides['mode'] in ['track', 'predict']
		overrides['save'] = kwargs.get('save', False)  # not save files by default
		if not self.predictor:
			self.task = overrides.get('task') or self.task
			self.predictor = TASK_MAP[self.task][3](overrides=overrides)
			self.predictor.setup_model(model=self.model, verbose=is_cli)
		else:  # only update args if predictor is already setup
			self.predictor.args = get_cfg(self.predictor.args, overrides)
		return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

	def track(self, source=None, stream=False, **kwargs):
		from ultralytics.tracker import register_tracker
		register_tracker(self)
		# ByteTrack-based method needs low confidence predictions as input
		conf = kwargs.get('conf') or 0.1
		kwargs['conf'] = conf
		kwargs['mode'] = 'track'
		return self.predict(source=source, stream=stream, **kwargs)

	@smart_inference_mode()
	def val(self, data=None, **kwargs):
		"""
		Validate a model on a given dataset .

		Args:
			data (str): The dataset to validate on. Accepts all formats accepted by yolo
			**kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
		"""
		overrides = self.overrides.copy()
		overrides['rect'] = True  # rect batches as default
		overrides.update(kwargs)
		overrides['mode'] = 'val'
		args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
		args.data = data or args.data
		if 'task' in overrides:
			self.task = args.task
		else:
			args.task = self.task
		if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
			args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
		args.imgsz = check_imgsz(args.imgsz, max_dim=1)

		validator = TASK_MAP[self.task][2](args=args)
		validator(model=self.model)
		self.metrics = validator.metrics

		return validator.metrics

	@smart_inference_mode()
	def benchmark(self, **kwargs):
		"""
		Benchmark a model on all export formats.

		Args:
			**kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
		"""
		self._check_is_pytorch_model()
		from ultralytics.yolo.utils.benchmarks import benchmark
		overrides = self.model.args.copy()
		overrides.update(kwargs)
		overrides = {**DEFAULT_CFG_DICT, **overrides}  # fill in missing overrides keys with defaults
		return benchmark(model=self, imgsz=overrides['imgsz'], half=overrides['half'], device=overrides['device'])

	def export(self, **kwargs):
		"""
		Export model.

		Args:
			**kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
		"""
		self._check_is_pytorch_model()
		overrides = self.overrides.copy()
		overrides.update(kwargs)
		args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
		args.task = self.task
		if args.imgsz == DEFAULT_CFG.imgsz:
			args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
		if args.batch == DEFAULT_CFG.batch:
			args.batch = 1  # default to 1 if not modified
		return Exporter(overrides=args)(model=self.model)

	def train(self, **kwargs):
		"""
		Trains the model on a given dataset.

		Args:
			**kwargs (Any): Any number of arguments representing the training configuration.
		"""
		self._check_is_pytorch_model()
		self._check_pip_update()
		overrides = self.overrides.copy()
		overrides.update(kwargs)
		if kwargs.get('cfg'):
			LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
			overrides = yaml_load(check_yaml(kwargs['cfg']))
		overrides['mode'] = 'train'
		if not overrides.get('data'):
			raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
		if overrides.get('resume'):
			overrides['resume'] = self.ckpt_path

		self.task = overrides.get('task') or self.task
		self.trainer = TASK_MAP[self.task][1](overrides=overrides)
		if not overrides.get('resume'):  # manually set model only if not resuming
			self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
			self.model = self.trainer.model
		self.trainer.hub_session = self.session  # attach optional HUB session
		self.trainer.train()
		# update model and cfg after training
		if RANK in {0, -1}:
			self.model, _ = attempt_load_one_weight(str(self.trainer.best))
			self.overrides = self.model.args
			self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP

	def to(self, device):
		"""
		Sends the model to the given device.

		Args:
			device (str): device
		"""
		self._check_is_pytorch_model()
		self.model.to(device)

	@property
	def names(self):
		"""
		 Returns class names of the loaded model.
		"""
		return self.model.names if hasattr(self.model, 'names') else None

	@property
	def device(self):
		"""
		Returns device if PyTorch model
		"""
		return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

	@property
	def transforms(self):
		"""
		 Returns transform of the loaded model.
		"""
		return self.model.transforms if hasattr(self.model, 'transforms') else None

	@staticmethod
	def add_callback(event: str, func):
		"""
		Add callback
		"""
		callbacks.default_callbacks[event].append(func)

	@staticmethod
	def _reset_ckpt_args(args):
		include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
		return {k: v for k, v in args.items() if k in include}

	@staticmethod
	def _reset_callbacks():
		for event in callbacks.default_callbacks.keys():
			callbacks.default_callbacks[event] = [callbacks.default_callbacks[event][0]]


# MARK: - Utils

def adjust_state_dict(state_dict: OrderedDict) -> OrderedDict:
	od = OrderedDict()
	for key, value in state_dict.items():
		new_key     = key.replace("module.", "")
		od[new_key] = value
	return od


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
	# Rescale coords (xyxy) from img1_shape to img0_shape
	if ratio_pad is None:  # calculate from img0_shape
		gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
		pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
	else:
		gain = ratio_pad[0][0]
		pad = ratio_pad[1]

	coords[:, [0, 2]] -= pad[0]  # x padding
	coords[:, [1, 3]] -= pad[1]  # y padding
	coords[:, :4] /= gain
	# clip_coords(coords, img0_shape)
	return coords
