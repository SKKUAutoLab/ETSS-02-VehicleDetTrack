#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
import uuid
import glob
from queue import Queue
from operator import itemgetter
from timeit import default_timer as timer
from typing import Union

import cv2
import torch
import numpy as np
from tqdm import tqdm

from core.data.class_label import ClassLabels
from core.io.filedir import is_basename
from core.io.filedir import is_json_file
from core.io.filedir import is_stem
from core.utils.bbox import bbox_xyxy_to_cxcywh_norm
from core.utils.rich import console
from core.utils.constants import AppleRGB
from core.io.frame import FrameLoader
from core.io.frame import FrameWriter
from core.io.video import is_video_file
from core.io.video import VideoLoader
from core.io.picklewrap import PickleLoader
from core.factory.builder import CAMERAS
from core.factory.builder import DETECTORS
from detectors.base import BaseDetector
from configuration import (
	data_dir,
	config_dir
)
from cameras.base import BaseCamera

__all__ = [
	"TrafficSafetyCamera"
]

# MARK: - TrafficSafetyCamera


# noinspection PyAttributeOutsideInit

@CAMERAS.register(name="traffic_safety_camera")
class TrafficSafetyCamera(BaseCamera):

	# MARK: Magic Functions

	def __init__(
			self,
			data         : dict,
			dataset      : str,
			name         : str,
			detector     : dict,
			identifier   : dict,
			data_loader  : dict,
			data_writer  : Union[FrameWriter,  dict],
			process      : dict,
			id_          : Union[int, str] = uuid.uuid4().int,
			verbose      : bool            = False,
			queue_size   : int = 10,
			*args, **kwargs
	):
		"""

		Args:
			dataset (str):
				Dataset name. It is also the name of the directory inside
				`data_dir`.
			subset (str):
				Subset name. One of: [`dataset_a`, `dataset_b`].
			name (str):
				Camera name. It is also the name of the camera's config
				files.
			class_labels (ClassLabels, dict):
				ClassLabels object or a config dictionary.
			detector (BaseDetector, dict):
				Detector object or a detector's config dictionary.
			data_loader (FrameLoader, dict):
				Data loader object or a data loader's config dictionary.
			data_writer (VideoWriter, dict):
				Data writer object or a data writer's config dictionary.
			id_ (int, str):
				Camera's unique ID.
			verbose (bool):
				Verbosity mode. Default: `False`.
			queue_size (int):
				Size of queue store the information
		"""
		super().__init__(id_=id_, dataset=dataset, name=name)
		# NOTE: Init attributes
		self.start_time = None
		self.pbar       = None

		# NOTE: Define attributes
		self.process      = process
		self.verbose      = verbose

		self.data_cfg        = data
		self.data_loader_cfg = data_loader
		self.data_writer_cfg = data_writer
		self.detector_cfg    = detector
		self.identifier_cfg  = identifier

		self.init_dirs()
		self.init_data_loader(data_loader_cfg=self.data_loader_cfg)
		self.init_data_writer(data_writer_cfg=self.data_writer_cfg)
		self.init_class_labels(class_labels=self.detector_cfg['class_labels'])
		self.init_detector(detector=detector)

		# NOTE: Queue
		self.frames_queue          = Queue(maxsize = self.data_loader_cfg['queue_size'])
		self.detections_queue      = Queue(maxsize = self.detector_cfg['queue_size'])
		self.identifications_queue = Queue(maxsize = self.identifier_cfg['queue_size'])

	# MARK: Configure

	def init_dirs(self):
		"""Initialize dirs.

		Returns:

		"""
		self.root_dir    = os.path.join(data_dir)
		self.configs_dir = os.path.join(config_dir)
		self.outputs_dir = os.path.join(self.root_dir, self.data_writer_cfg["dst"])
		self.video_dir   = os.path.join(self.root_dir, self.data_loader_cfg["data"])
		self.image_dir   = os.path.join(self.root_dir, self.data_loader_cfg["data"])

	def init_class_labels(self, class_labels: Union[ClassLabels, dict]):
		"""Initialize class_labels.

		Args:
			class_labels (ClassLabels, dict):
				ClassLabels object or a config dictionary.
		"""
		if isinstance(class_labels, ClassLabels):
			self.class_labels = class_labels
		elif isinstance(class_labels, dict):
			file = class_labels["file"]
			if is_json_file(file):
				self.class_labels = ClassLabels.create_from_file(file)
			elif is_basename(file):
				file              = os.path.join(self.root_dir, file)
				self.class_labels = ClassLabels.create_from_file(file)
		else:
			file              = os.path.join(self.root_dir, f"class_labels.json")
			self.class_labels = ClassLabels.create_from_file(file)
			print(f"Cannot initialize class_labels from {class_labels}. "
				  f"Attempt to load from {file}.")

	def init_detector(self, detector: Union[BaseDetector, dict]):
		"""Initialize detector.

		Args:
			detector (BaseDetector, dict):
				Detector object or a detector's config dictionary.
		"""
		console.log(f"Initiate Detector.")
		if isinstance(detector, BaseDetector):
			self.detector = detector
		elif isinstance(detector, dict):
			detector["class_labels"] = self.class_labels
			self.detector = DETECTORS.build(**detector)
		else:
			raise ValueError(f"Cannot initialize detector with {detector}.")

	def init_data_loader(self, data_loader_cfg: dict):
		"""Initialize data loader.

		Args:
			data_loader_cfg (dict):
				Data loader object or a data loader's config dictionary.
		"""
		if self.process["run_image"]:
			self.data_loader = FrameLoader(data=os.path.join(data_dir, data_loader_cfg["data_path"]), batch_size=data_loader_cfg["batch_size"])
		else:
			self.data_loader = VideoLoader(data=os.path.join(data_dir, data_loader_cfg["data_path"]), batch_size=data_loader_cfg["batch_size"])

		self.pbar = tqdm(total=self.data_loader.num_frames, desc=f"{data_loader_cfg['data_path']}")

	def check_and_create_folder(self, attr, data_writer_cfg: dict):
		"""CHeck and create the folder to store the result

		Args:
			attr (str):
				the type of function/saving/creating
			data_writer_cfg (dict):
				configuration of camera
		Returns:
			None
		"""
		path = os.path.join(self.outputs_dir, f"{data_writer_cfg[attr]}")
		if not os.path.isdir(path):
			os.makedirs(path)
		data_writer_cfg[attr] = path

	def init_data_writer(self, data_writer_cfg: dict):
		"""Initialize data writer.

		Args:
			data_writer_cfg (FrameWriter, dict):
				Data writer object or a data writer's config dictionary.
		"""
		# NOTE: save detections crop
		data_writer_cfg["dets_crop_pkl"] = f'{data_writer_cfg["dets_crop_pkl"]}/{self.detector_cfg["folder_out"]}'
		self.check_and_create_folder("dets_crop_pkl", data_writer_cfg=data_writer_cfg)

	# MARK: Run

	def run_detector(self):
		"""Run detection model with videos
		"""
		# init value
		height_img, width_img = None, None

		# NOTE: run detection
		with torch.no_grad():  # phai them cai nay khong la bi memory leak
			for images, indexes, _, _ in self.data_loader:
				# NOTE: pre process
				# if finish loading
				if len(indexes) == 0:
					break

				# get size of image
				if height_img is None:
					height_img, width_img, _ = images[0].shape

				# NOTE: Detect batch of instances
				batch_instances = self.detector.detect(
					indexes=indexes, images=images
				)

				# NOTE: Write the detection result
				for index_b, (index_image, batch) in enumerate(zip(indexes, batch_instances)):
				# for index_b, batch in enumerate(batch_instances):
					# DEBUG:
					# image_draw = images[index_b].copy()

					# store result each frame
					batch_detections = []

					for index_in, instance in enumerate(batch):
						name_index_image = f"{index_image:08d}_{index_in:08d}"
						bbox_xyxy = [int(i) for i in instance.bbox]

						# if size of bounding box is very small
						# because the heuristic need the bigger bounding box
						if abs(bbox_xyxy[2] - bbox_xyxy[0]) < 40 \
								or abs(bbox_xyxy[3] - bbox_xyxy[1]) < 40:
							continue

						# NOTE: crop the bounding box, add 60 or 1.5 scale
						bbox_xyxy  = scaleup_bbox(bbox_xyxy, height_img, width_img, ratio=1.5, padding=60)
						crop_image = images[index_b][bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]

						# DEBUG:
						# if instance.confidence < 0.1:
						# 	continue
						# print(bbox_xyxy)
						# cv2.rectangle(image_draw, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (125, 125, 125), 4, cv2.LINE_AA)  # filled

						detection_result = {
							'video_name': self.data_loader_cfg['data_path'],
							'frame_id'  : index_image,
							'crop_id'   : index_in,
							'crop_img'  : crop_image,
							'bbox'      : (bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]),
							'class_id'  : instance.class_label["train_id"],
							'id'        : instance.class_label["id"],
							'conf'      : instance.confidence,
							'width_img' : width_img,
							'height_img': height_img
						}
						batch_detections.append(detection_result)

					# NOTE: Push detections to queue
					self.detections_queue.put([indexes, batch_detections])

					# DEBUG:
					# cv2.imwrite(
					# 	f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5_test_docker/output_aic23/dets_crop_debug/001/" \
					# 	f"{name_index_image}.jpg",
					# 	image_draw
					# )

				# self.pbar.update(len(indexes))

		# NOTE: Push None to queue to act as a stopping condition for next thread
		self.detections_queue.put([None, None])

	def run_identifier(self):
		"""Run detection model

		Returns:

		"""
		# NOTE: Run identification
		with torch.no_grad():  # phai them cai nay khong la bi memory leak
			while True:
				# NOTE: Get batch detections from queue
				(indexes, batch_detections) = self.detections_queue.get()

				if batch_detections is None:
					break

				for idx, detection_result in enumerate(batch_detections):
					crop_images = []
					# # Load crop images
					# for pkl in pickles:
					# 	crop_images.append(pkl['crop_img'])
					#
					# # NOTE: Identify batch of instances
					# batch_instances = self.identifier.detect(
					# 	indexes=indexes, images=crop_images
					# )
					#
					# # NOTE: Write the full detection result
					# for index_b, (crop_dict, batch) in enumerate(zip(pickles, batch_instances)):
					# 	for index_in, instance in enumerate(batch):
					# 		bbox_xyxy     = [int(i) for i in instance.bbox]
					#
					# 		# NOTE: add the coordinate from crop image to original image
					# 		# DEBUG: comment doan nay neu extract anh nho
					# 		bbox_xyxy[0] += int(crop_dict["bbox"][0])
					# 		bbox_xyxy[1] += int(crop_dict["bbox"][1])
					# 		bbox_xyxy[2] += int(crop_dict["bbox"][0])
					# 		bbox_xyxy[3] += int(crop_dict["bbox"][1])
					#
					# 		# if size of bounding box is very small
					# 		if abs(bbox_xyxy[2] - bbox_xyxy[0]) < 40 \
					# 				or abs(bbox_xyxy[3] - bbox_xyxy[1]) < 40:
					# 			continue
					#
					# 		result_dict = {
					# 			'video_name': crop_dict['video_name'],
					# 			'frame_id'  : crop_dict['frame_id'],
					# 			'crop_id'   : index_b,
					# 			'bbox'      : (bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]),
					# 			'class_id'  : instance.class_label["train_id"],
					# 			'id'        : instance.class_label["id"],
					# 			'conf'      : (float(crop_dict["conf"]) * instance.confidence),
					# 			'width_img' : crop_dict['width_img'],
					# 			'height_img': crop_dict['height_img']
					# 		}
					# 		out_dict.append(result_dict)

				self.pbar.update(len(indexes))

	def run(self):
		"""Main run loop."""
		self.run_routine_start()

		# NOTE: run detector
		self.run_detector()
		self.detector.clear_model_memory()
		self.detector = None

		# NOTE: run identification
		self.run_identifier()

		self.run_routine_end()

	def run_routine_start(self):
		"""Perform operations when run routine starts. We start the timer."""
		self.start_time = timer()
		if self.verbose:
			cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)

	def run_routine_end(self):
		"""Perform operations when run routine ends."""
		cv2.destroyAllWindows()
		self.stop_time = timer()

	def postprocess(self, image: np.ndarray, *args, **kwargs):
		"""Perform some postprocessing operations when a run step end.

		Args:
			image (np.ndarray):
				Image.
		"""
		if not self.verbose and not self.save_image and not self.save_video:
			return

		elapsed_time = timer() - self.start_time
		if self.verbose:
			# cv2.imshow(self.name, result)
			cv2.waitKey(1)

	# MARK: Visualize

	def draw(self, drawing: np.ndarray, elapsed_time: float) -> np.ndarray:
		"""Visualize the results on the drawing.

		Args:
			drawing (np.ndarray):
				Drawing canvas.
			elapsed_time (float):
				Elapsed time per iteration.

		Returns:
			drawing (np.ndarray):
				Drawn canvas.
		"""
		return drawing


# MARK - Ultilies

def scaleup_bbox(bbox_xyxy, height_img, width_img, ratio, padding):
	"""Scale up 1.2% or +-40

	Args:
		bbox_xyxy:
		height_img:
		width_img:

	Returns:

	"""
	cx = 0.5 * bbox_xyxy[0] + 0.5 * bbox_xyxy[2]
	cy = 0.5 * bbox_xyxy[1] + 0.5 * bbox_xyxy[3]
	w = abs(bbox_xyxy[2] - bbox_xyxy[0])
	w = min(w * ratio, w + padding)
	h = abs(bbox_xyxy[3] - bbox_xyxy[1])
	h = min(h * ratio, h + padding)
	bbox_xyxy[0] = int(max(0, cx - 0.5 * w))
	bbox_xyxy[1] = int(max(0, cy - 0.5 * h))
	bbox_xyxy[2] = int(min(width_img - 1, cx + 0.5 * w))
	bbox_xyxy[3] = int(min(height_img - 1, cy + 0.5 * h))
	return bbox_xyxy
