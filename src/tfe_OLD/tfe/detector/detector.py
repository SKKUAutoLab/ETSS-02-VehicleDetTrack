# ==================================================================== #
# File name: tracker.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/27/2021
# ==================================================================== #
import abc
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch.nn as nn
from munch import Munch
from torch import Tensor

from .detection import Detection
from tfe.utils import data_dir
from tfe.utils import parse_config_from_json
from tfe.utils import printe
from tfe.utils import printw
from tfe.utils import select_device


# MARK: - Detector

class Detector(metaclass=abc.ABCMeta):
	"""Detector Base Class.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		api            : Optional[str]                  = None,
		name           : Optional[str]                  = None,
		variant        : Optional[str]                  = None,
		weights        : Optional[str]                  = None,
		dataset        : Optional[str]                  = None,
		dims           : Optional[Tuple[int, int, int]] = None,
		min_confidence : float                          = 0.5,
		nms_max_overlap: float                          = 0.4,
		device         : Optional[str]                  = None,
		labels         : Union[List[Dict], None]        = None,
		**kwargs
	):
		# NOTE: Define hyperparameters
		super().__init__(**kwargs)
		self.api             = api
		self.name            = name
		self.variant         = variant if variant is not None else name
		self.weights         = weights
		self.dataset         = dataset
		self.dims            = dims
		self.min_confidence  = min_confidence
		self.nms_max_overlap = nms_max_overlap
		self.model           = None
		self.device          = select_device(device=device)
		
		# NOTE: NEVER LOAD MODEL HERE
		
		# NOTE: Load the labels from ``config.aic_labels.json``.
		if labels is None:
			dataset_dir    = os.path.join(data_dir, self.dataset)
			labels         = parse_config_from_json(json_path=os.path.join(dataset_dir, "labels.json"))
			labels         = Munch.fromDict(labels)
			self.labels    = labels.labels
		else:
			self.labels    = labels
		self.label_ids = [label.train_id for label in self.labels]
		
	# MARK: Detection
	
	def detect_objects(
		self,
		frame_indexes: int,
		images       : Union[Tensor, np.ndarray]
	) -> Union[List[Detection], List[List[Detection]]]:
		"""Detect road_objects in the image.
		"""
		# NOTE: Safety check
		if self.model is None:
			printe("Model has not been defined yet!")
			raise NotImplementedError
	
		# NOTE: Forward Pass
		batch_detections = self.forward_pass(frame_indexes=frame_indexes, images=images)
		
		# NOTE: Check allowed labels
		[self.suppress_wrong_labels(detections=detections) for detections in batch_detections]
		
		# NOTE: Suppress confident score
		[self.suppress_low_confident(detections=detections) for detections in batch_detections]
		
		# NOTE: Suppress NMS
		[self.non_max_suppression(detections=detections) for detections in batch_detections]
		
		return batch_detections
	
	def prepare_input(self, images: Union[Tensor, np.ndarray]) -> Tensor:
		"""Prepare the model's input for the forward pass.
		"""
		printw("Prepare input has not been implemented yet")
		pass
	
	def forward_pass(
		self,
		frame_indexes: List[int],
		images       : Union[Tensor, np.ndarray]
	) -> Union[List[Detection], List[List[Detection]]]:
		"""Define the forward pass logic of the ``model``.
		"""
		printe("Forward pass has not been implemented yet")
		raise NotImplementedError
		
	def suppress_wrong_labels(self, detections: List):
		"""Suppress any wrong labels.
		"""
		valid_detections = [d for d in detections if self.is_correct_label(d.label)]
		return valid_detections
	
	def is_correct_label(self, label: Optional[Dict]):
		"""Check if the label is allowed in our application.
		"""
		if label.train_id in self.label_ids:
			return True
		return False
	
	def suppress_low_confident(self, detections: List):
		"""Suppress detections of low-confidence.
		"""
		valid_detections = [d for d in detections if d.confidence >= self.min_confidence]
		return valid_detections
	
	def non_max_suppression(self, detections: List):
		"""Suppress overlapping detections (high-level).
		"""
		# NOTE: Extract detection bounding boxes and scores
		boxes  = np.array([d.bbox       for d in detections])
		scores = np.array([d.confidence for d in detections])
		
		# NOTE: Extract road_objects indices that survive non-max-suppression
		indices = []
		if len(boxes) > 0:
			boxes = boxes.astype(np.float)
			
			# Top-left to Bottom-right
			x1 = boxes[:, 0]
			y1 = boxes[:, 1]
			x2 = boxes[:, 2] + boxes[:, 0]
			y2 = boxes[:, 3] + boxes[:, 1]
			
			# Area
			area = (boxes[:, 2] + 1) * (boxes[:, 3] + 1)
			if scores is not None:
				idxs = np.argsort(scores)
			else:
				idxs = np.argsort(y2)
			
			# Suppression via iterating boxes
			while len(idxs) > 0:
				last = len(idxs) - 1
				i = idxs[last]
				indices.append(i)
				
				xx1 = np.maximum(x1[i], x1[idxs[:last]])
				yy1 = np.maximum(y1[i], y1[idxs[:last]])
				xx2 = np.minimum(x2[i], x2[idxs[:last]])
				yy2 = np.minimum(y2[i], y2[idxs[:last]])
				
				w = np.maximum(0, xx2 - xx1 + 1)
				h = np.maximum(0, yy2 - yy1 + 1)
				
				overlap = (w * h) / area[idxs[:last]]
				
				idxs = np.delete(
					idxs, np.concatenate(
						([last], np.where(overlap > self.nms_max_overlap)[0])))
		
		# NOTE: Get exactly the vehicles surviving non-max-suppression
		detections = [detections[i] for i in indices]
		return detections
