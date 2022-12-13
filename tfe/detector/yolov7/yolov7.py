# ==================================================================== #

# ==================================================================== #
import os
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
		ckpt = torch.load(self.weights)  # load
		model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())

		# NOTE: Compatibility updates
		for m in model.modules():
			if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
				m.inplace = True  # pytorch 1.7.0 compatibility
			elif type(m) is nn.Upsample:
				m.recompute_scale_factor = None  # torch 1.11.0 compatibility
			elif type(m) is Conv:
				m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

# MARK: Detection


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
