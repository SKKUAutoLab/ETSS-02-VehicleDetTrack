# ==================================================================== #
# Copyright (C) 2022 - Automation Lab - Sungkyunkwan University
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# ==================================================================== #
from __future__ import annotations

import os
import pickle
import sys
import threading
import uuid
import glob
import copy
import random
from queue import Queue
from operator import itemgetter
from timeit import default_timer as timer
from typing import Union

import cv2
import torch
import numpy as np
from tqdm import tqdm

from analyses.analyzer import BaseAnalyzer
from core.data.class_label import ClassLabels
from core.io.filedir import is_basename
from core.io.filedir import is_json_file
from core.io.frame import FrameLoader
from core.io.frame import FrameWriter
from core.io.picklewrap import PickleLoader
from core.io.video import is_video_file
from core.io.video import VideoLoader
from core.objects.general_object import GeneralObject
from core.objects.gmo import GMO
from core.objects.moving_model import MovingState
from core.utils.bbox import bbox_xyxy_to_cxcywh_norm
from core.utils.label import get_label
from core.utils.rich import console
from core.utils.constants import AppleRGB
from core.objects.instance import Instance
from core.factory.builder import CAMERAS, TRACKERS, MATCHERS, ANALYZERS
from core.factory.builder import DETECTORS
from detectors.detector import BaseDetector
from trackers.tracker import Tracker
from matcher.matcher import BaseMatcher
from matcher.moi import MOI
from matcher.roi import ROI

from configuration import (
	data_dir,
	config_dir,
	result_dir
)
from cameras.base import BaseCamera

__all__ = [
	"TrafficSafetyCamera"
]


# NOTE: only for ACI23_Track_5
classes_aic23 = ['motorbike',
				 'DHelmet', 'DNoHelmet',
				 'P1Helmet', 'P1NoHelmet',
				 'P2Helmet', 'P2NoHelmet']

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
			tracker      : dict,
			matcher      : dict,
			analyzer     : dict,
			data_loader  : dict,
			data_writer  : Union[FrameWriter,  dict],
			process      : dict,
			id_          : Union[int, str] = uuid.uuid4().int,
			verbose      : bool            = False,
			queue_size   : int             = 10,
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
		self.detector   = None
		self.identifier = None

		# NOTE: Define attributes
		self.process         = process
		self.verbose         = verbose

		# NOTE: Define configurations
		self.data_cfg        = data
		self.detector_cfg    = detector
		self.identifier_cfg  = identifier
		self.tracker_cfg     = tracker
		self.matcher_cfg     = matcher
		self.analyzer_cfg    = analyzer
		self.data_loader_cfg = data_loader
		self.data_writer_cfg = data_writer

		# NOTE: Queue
		self.frames_queue                 = Queue(maxsize = self.data_loader_cfg['queue_size'])
		self.detections_queue_identifier  = Queue(maxsize = self.detector_cfg['queue_size'])
		self.detections_queue_tracker     = Queue(maxsize = self.detector_cfg['queue_size'])
		self.identifications_queue        = Queue(maxsize = self.identifier_cfg['queue_size'])
		self.trackings_queue              = Queue(maxsize = self.tracker_cfg['queue_size'])
		self.matching_queue               = Queue(maxsize = self.matcher_cfg['queue_size'])
		self.writer_queue                 = Queue(maxsize = self.data_writer_cfg['queue_size'])

		# NOTE: Init modules
		self.init_dirs()
		self.init_data_loader(data_loader_cfg = self.data_loader_cfg)
		self.init_class_labels(class_labels   = self.detector_cfg['class_labels'])
		self.init_detector(detector           = detector)
		self.init_identifier(identifier       = identifier)
		self.init_tracker(tracker             = tracker)
		self.init_matcher(matcher             = matcher)
		self.init_analyzer(analyzer           = analyzer)
		self.init_gmo(matcher                 = matcher)

		# NOTE: Init for output
		self.init_data_output()

	# MARK: Configure

	def init_dirs(self):
		"""Initialize dirs.

		Returns:

		"""
		self.root_dir    = os.path.join(data_dir)
		self.configs_dir = os.path.join(config_dir)
		self.outputs_dir = os.path.join(result_dir, self.data_writer_cfg["dst"])
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

	def init_identifier(self, identifier: Union[BaseDetector, dict]):
		"""Initialize identifier.

		Args:
			identifier (BaseDetector, dict):
				Identifier object or a identifier's config dictionary.
		"""
		console.log(f"Initiate Identifier.")
		if isinstance(identifier, BaseDetector):
			self.identifier = identifier
		elif isinstance(identifier, dict):
			identifier["class_labels"] = self.class_labels
			self.identifier = DETECTORS.build(**identifier)
		else:
			raise ValueError(f"Cannot initialize detector with {identifier}.")

	def init_tracker(self, tracker: Union[Tracker, dict]):
		"""Initialize tracker.

		Args:
			tracker (Tracker, dict):
				Tracker object or a tracker's config dictionary.
		"""
		console.log(f"Initiate Tracker.")
		if isinstance(tracker, Tracker):
			self.tracker = tracker
		elif isinstance(tracker, dict):
			self.tracker = TRACKERS.build(**tracker)
		else:
			raise ValueError(f"Cannot initialize detector with {tracker}.")

	def init_matcher(self, matcher: dict):
		"""Initialize matcher.

		Args:
			matcher (dict):
				Matcher object or a matcher's config dictionary.
		"""
		# self.rois = ROI.load_rois_from_file(dataset=dataset, file=file, **hparams)
		# self.mois = MOI.load_mois_from_file(dataset=dataset, file=file, **hparams)
		console.log(f"Initiate Matcher.")
		if isinstance(matcher, BaseMatcher):
			self.matcher = matcher
		elif isinstance(matcher, dict):
			self.matcher = MATCHERS.build(**matcher)
		else:
			raise ValueError(f"Cannot initialize detector with {matcher}.")

	def init_analyzer(self, analyzer: dict):
		"""Initialize tracker.

		Args:
			analyzer (dict):
				Analyzer object or a analyzer's config dictionary.
		"""
		console.log(f"Initiate Analyzer.")
		if isinstance(analyzer, BaseAnalyzer):
			self.analyzer = analyzer
		elif isinstance(analyzer, dict):
			self.analyzer = ANALYZERS.build(**analyzer)
		else:
			raise ValueError(f"Cannot initialize detector with {analyzer}.")

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

	def init_gmo(self, matcher: dict):
		GeneralObject.min_travelled_distance = matcher["gmo"]["min_traveled_distance"]
		GMO.min_traveled_distance            = matcher["gmo"]["min_traveled_distance"]
		GMO.min_entering_distance            = matcher["gmo"]["min_entering_distance"]
		GMO.min_hit_streak                   = matcher["gmo"]["min_hit_streak"]
		GMO.max_age                          = matcher["gmo"]["max_age"]

	def check_and_create_folder(self, folder_path):
		"""Check and create the folder to store the result

		Args:
			folder_path (str):
				path to folder
		Returns:
			None
		"""
		if not os.path.isdir(folder_path):
			os.makedirs(folder_path)

	def init_data_output(self):
		"""Initialize data writer.
		"""

		# NOTE: save detections
		self.data_writer_cfg["detector"]  = os.path.join(self.outputs_dir, "detector", self.detector_cfg['folder_out'])
		self.check_and_create_folder(self.data_writer_cfg["detector"])

		# NOTE: save identifier
		self.data_writer_cfg["identifier"] = os.path.join(self.outputs_dir, "identifier", self.identifier_cfg['folder_out'])
		self.check_and_create_folder(self.data_writer_cfg["identifier"])

		# NOTE: save matcher
		self.data_writer_cfg["matcher"] = os.path.join(self.outputs_dir, "matcher", self.matcher_cfg['folder_out'])
		self.check_and_create_folder(self.data_writer_cfg["matcher"])

		# NOTE: save analyzer
		self.data_writer_cfg["analyzer"] = os.path.join(self.outputs_dir, "analyzer", self.analyzer_cfg['folder_out'])
		self.check_and_create_folder(self.data_writer_cfg["analyzer"])

	# MARK: Run

	def run_detector(self):
		"""Run detection model with videos"""
		# NOTE: create directory to store result
		folder_output = f"{self.data_writer_cfg['detector']}/{self.data_loader_cfg['data_path']}/detection/"
		make_directory(folder_output)

		# init value
		height_img, width_img = None, None

		# NOTE: init data loader for load images/video
		self.init_data_loader(self.data_loader_cfg)
		detections_queue_identifier = []
		detections_queue_tracker    = []

		pbar = tqdm(total=self.data_loader.num_frames, desc=f"Detection: ")

		# NOTE: run detection
		with torch.no_grad():  # phai them cai nay khong la bi memory leak
			for images, indexes, _, _ in self.data_loader:

				if len(indexes) == 0:
					break

				# if finish loading
				if indexes is None:
					break

				# get size of image
				if height_img is None:
					height_img, width_img, _ = images[0].shape

				# NOTE: Detect batch of instances
				batch_instances = self.detector.detect(
					indexes=indexes, images=images
				)

				for index_b, (index_image, batch) in enumerate(zip(indexes, batch_instances)):
					# NOTE: Associate detections with ROI (in batch)
					ROI.associate_detections_to_rois(detections=batch, rois=self.matcher.rois)
					batch_instances[index_b] = [d for d in batch if d.roi_uuid is not None]

				# NOTE: Process the detection result
				for index_b, (index_image, batch) in enumerate(zip(indexes, batch_instances)):
				# for index_b, batch in enumerate(batch_instances):
					# store result each frame
					batch_detections_identifier = []
					batch_detections_tracker    = []

					# DEBUG: draw
					image_draw = images[index_b].copy()

					# NOTE: Process each detection
					for index_in, instance in enumerate(batch):
						bbox_xyxy = [int(i) for i in instance.bbox]
						crop_id   = [int(index_image), int(index_in)]  # frame_index, bounding_box index

						# if size of bounding box is very small
						# because the heuristic need the bigger bounding box
						if abs(bbox_xyxy[2] - bbox_xyxy[0]) < 40 \
								or abs(bbox_xyxy[3] - bbox_xyxy[1]) < 40:
							continue

						# NOTE: crop the bounding box for identifier, add 60 or 1.5 scale
						bbox_xyxy  = scaleup_bbox(
							bbox_xyxy,
							height_img,
							width_img,
							ratio   = 1.5,
							padding = 60
						)
						detection_result = {
							'roi_uuid'    : instance.roi_uuid,
							'video_name'  : self.data_loader_cfg['data_path'],
							'frame_index' : index_image,
							'image'       : images[index_b][bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]],
							'bbox'        : np.array((bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3])),
							'class_id'    : instance.class_label["train_id"],
							'class_label' : instance.class_label,
							'label'       : instance.label,
							'id'          : crop_id,
							'confidence'  : instance.confidence,
							'image_size'  : [width_img, height_img]
						}
						# detection_instance = Instance(**detection_result)
						batch_detections_identifier.append(Instance(**detection_result))

						# DEBUG: draw
						image_draw = plot_one_box(
							bbox = bbox_xyxy,
							img  = image_draw,
							label= instance.label.name
						)

						# NOTE: crop the bounding box for tracker, add 40 or 1.2 scale
						bbox_xyxy = [int(i) for i in instance.bbox]
						bbox_xyxy = scaleup_bbox(
							bbox_xyxy,
							height_img,
							width_img,
							ratio   = 1.0,
							padding = 0
						)
						detection_result = {
							'roi_uuid'    : instance.roi_uuid,
							'video_name'  : self.data_loader_cfg['data_path'],
							'frame_index' : index_image,
							'image'       : images[index_b][bbox_xyxy[1]: bbox_xyxy[3], bbox_xyxy[0]: bbox_xyxy[2]],
							'bbox'        : np.array((bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3])),
							'class_id'    : instance.class_label["train_id"],
							'class_label' : instance.class_label,
							'label'       : instance.label,
							'id'          : crop_id,
							'confidence'  : instance.confidence,
							'image_size'  : [width_img, height_img]
						}
						batch_detections_tracker.append(Instance(**detection_result))

					# DEBUG: draw
					cv2.imwrite(os.path.join(folder_output, f"{index_image:04d}.jpg"), image_draw)

					# NOTE: Push detections to array
					detections_queue_identifier.append([index_image, images[index_b], batch_detections_identifier])
					detections_queue_tracker.append([index_image, images[index_b], batch_detections_tracker])

				# update pbar
				pbar.update(len(indexes))

		pbar.close()

		# NOTE: save pickle
		pickle.dump(
			detections_queue_identifier,
			open(f"{self.data_writer_cfg['detector']}/{self.data_loader_cfg['data_path']}/detections_queue_identifier.pkl", 'wb')
		)
		pickle.dump(
			detections_queue_tracker,
			open(f"{self.data_writer_cfg['detector']}/{self.data_loader_cfg['data_path']}/detections_queue_tracker.pkl", 'wb')
		)


	def run_identifier(self):
		"""Run identification model"""
		# NOTE: create directory to store result
		folder_output = f"{self.data_writer_cfg['identifier']}/{self.data_loader_cfg['data_path']}/identification/"
		make_directory(folder_output)

		# NOTE: init
		pickle_loader = PickleLoader(
			data=f"{self.data_writer_cfg['detector']}/{self.data_loader_cfg['data_path']}/detections_queue_identifier.pkl",
			batch_size=self.identifier_cfg["batch_size"]
		)
		identifications_queue = []

		pbar = tqdm(total=len(pickle_loader), desc=f"Identification: ")

		# NOTE: Run identification
		with torch.no_grad():  # phai them cai nay khong la bi memory leak
			for pickles, indexes_img in pickle_loader:

				for index_frame, frame, batch_detections in pickles:
					# DEBUG: draw
					image_draw = frame.copy()

					# Load crop images
					crop_images = []
					indexes     = []
					for detection_instance in batch_detections:
						crop_images.append(detection_instance.image)
						indexes.append(detection_instance.id[1])

					# NOTE: Identify batch of instances
					if len(indexes) > 0 and len(crop_images) > 0:
						batch_instances = self.identifier.detect(
							indexes=indexes, images=crop_images
						)
					else:
						continue

					# store result each crop image
					batch_identifications = []

					# NOTE: Process the full identify result
					for index_b, (detection_instance, batch_instance) in enumerate(zip(batch_detections, batch_instances)):
						for index_in, instance in enumerate(batch_instance):
							bbox_xyxy     = [int(i) for i in instance.bbox]
							instance_id   = detection_instance.id + [int(index_in)]  # frame_index, bounding_box index, instance_index

							# NOTE: add the coordinate from crop image to original image
							# DEBUG: comment doan nay neu extract anh nho
							bbox_xyxy[0] += int(detection_instance.bbox[0])
							bbox_xyxy[1] += int(detection_instance.bbox[1])
							bbox_xyxy[2] += int(detection_instance.bbox[0])
							bbox_xyxy[3] += int(detection_instance.bbox[1])

							# if size of bounding box0 is very small
							if abs(bbox_xyxy[2] - bbox_xyxy[0]) < 40 \
									or abs(bbox_xyxy[3] - bbox_xyxy[1]) < 40:
								continue

							# NOTE: crop the bounding box, add 60 or 1.5 scale
							# bbox_xyxy = scaleup_bbox(
							# 	bbox_xyxy,
							# 	detection_instance.image_size[1],
							# 	detection_instance.image_size[0],
							# 	ratio   = 2.0,
							# 	padding = 60
							# )
							# DEBUG: draw
							image_draw = plot_one_box(
								bbox = bbox_xyxy,
								img  = image_draw,
								color= AppleRGB.values()[instance.class_label["train_id"]],
								label= instance.label.name
							)

							instance_image = frame[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]

							identification_result = {
								'video_name'    : detection_instance.video_name,
								'frame_index'   : detection_instance.frame_index,
								'bbox'          : np.array((bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3])),
								'class_id'      : instance.class_label["train_id"],
								'id'            : instance_id,
								'confidence'    : (float(detection_instance.confidence) * instance.confidence),
								'image_size'    : detection_instance.image_size
							}

							identification_instance = Instance(**identification_result)
							batch_identifications.append(identification_instance)

					# DEBUG: draw
					cv2.imwrite(os.path.join(folder_output, f"{index_frame:04d}.jpg"), image_draw)

					# NOTE: Push identifications to array
					identifications_queue.append([index_frame, frame, batch_identifications])

				pbar.update(len(indexes_img))

		pbar.close()

		# NOTE: save pickle
		pickle.dump(
			identifications_queue,
			open(f"{self.data_writer_cfg['identifier']}/{self.data_loader_cfg['data_path']}/identifications_queue.pkl", 'wb')
		)

	def run_tracker_matching(self):
		"""Run tracking"""
		# NOTE: create directory to store result
		folder_output = f"{self.data_writer_cfg['matcher']}/{self.data_loader_cfg['data_path']}/tracks_matching/"
		make_directory(folder_output)

		# NOTE: init
		pickle_loader = PickleLoader(
			data=f"{self.data_writer_cfg['detector']}/{self.data_loader_cfg['data_path']}/detections_queue_tracker.pkl",
			batch_size=self.tracker_cfg["batch_size"]
		)
		trackings_queue = []
		pbar = tqdm(total=len(pickle_loader), desc=f"Tracking: ")

		for pickles, indexes_img in pickle_loader:
			for index_frame, frame, batch_detections in pickles:

				# NOTE: Track (in batch)
				self.tracker.update(detections=batch_detections)
				gmos = self.tracker.tracks

				# NOTE: Update gmos by matcher
				self.matcher.update(gmos)
				self.matcher.update_lane(gmos)

				# NOTE: Push tracking to array
				trackings_queue.append([index_frame, frame, copy.deepcopy(gmos)])

				# DEBUG: Draw tracking and matching
				image_draw = frame.copy()
				# for gmo in gmos:
				# 	plot_one_box(
				# 		bbox = gmo.bboxes[-1],
				# 		img  = image_draw,
				# 		color= AppleRGB.values()[gmo.current_label.train_id],
				# 		label= f"{classes_aic23[gmo.current_label.train_id]}::{gmo.id % 1000}"
				# 	)
				image_draw = self.draw(
					drawing = image_draw,
					gmos    = gmos,
					rois    = self.matcher.rois,
					mois    = self.matcher.mois,
				)
				cv2.imwrite(os.path.join(folder_output, f"{index_frame:04d}.jpg"),image_draw)

			pbar.update(len(indexes_img))

		pbar.close()

		# NOTE: save pickle
		pickle.dump(
			trackings_queue,
			open(f"{self.data_writer_cfg['matcher']}/{self.data_loader_cfg['data_path']}/trackings_matching_queue.pkl", 'wb')
		)

	def run_analysis(self):
		"""Run tracking"""
		# NOTE: create directory to store result
		folder_output = f"{self.data_writer_cfg['analyzer']}/{self.data_loader_cfg['data_path']}/motorbike_violation/"
		make_directory(folder_output)

		# NOTE: load picker
		matching_pickle = pickle.load(
			open(f"{self.data_writer_cfg['matcher']}/{self.data_loader_cfg['data_path']}/trackings_matching_queue.pkl", 'rb'))

		identifications_pickle = pickle.load(
			open(f"{self.data_writer_cfg['identifier']}/{self.data_loader_cfg['data_path']}/identifications_queue.pkl", 'rb'))

		# NOTE: load in list
		matching_list = []
		for index_frame_match, frame, gmos in matching_pickle:
			matching_list.append([index_frame_match, frame, gmos])

		identifications_list = []
		for index_frame_ident, _, batch_identifications in identifications_pickle:
			identifications_list.append([index_frame_ident, _, batch_identifications])

		# NOTE: sync data frame by frame, trajectory by bounding boxes
		for index_frame_match, _, gmos in tqdm(matching_list, desc="Sync"):  # load all matching pickle
			for gmo in gmos:  # surf all gmo
				# if gmo.moving_state == MovingState.Counted:  # Only the counted object
				instances_in_bboxes = []

				for bboxe_id in gmo.bboxes_id:  # surf all bounding boxes in trajectory

					instances_in_bbox = []
					for index_frame_ident, _, batch_identifications in identifications_list:  # load all identification
						# if index_frame_match == index_frame_ident:  # check frame is the same or not
						for identification_instance in batch_identifications:  # surf all instances in the same frame_index
							if (bboxe_id[0] == identification_instance.id[0] and
									bboxe_id[1] == identification_instance.id[1]):
								instances_in_bbox.append(identification_instance)

					# if len(instances_in_bbox) > 0:  # if we get the people in the bounding box
					instances_in_bboxes.append(instances_in_bbox)

				# NOTE: update instances (people) in trajectory
				gmo.instances_in_bboxes = instances_in_bboxes

		# NOTE: analyse the violation
		gmo_violated_dict = {}
		gmo_violated_ids  = []
		for index_frame_match, _, gmos in tqdm(matching_list, desc="Analyze "):  # load all matching pickle

			# NOTE: analyse the violation on the motorbike
			# self.matcher.update_lane(gmos)
			self.analyzer.update(gmos)

			# NOTE: store the violation from GMOs
			for gmo in gmos:
				if gmo.moving_state in [MovingState.Counting, MovingState.ToBeCounted, MovingState.Counted, MovingState.Exiting]:
					if gmo.is_violated_num_people  \
							or gmo.is_violated_movement  \
							or gmo.is_violated_helmet:
						gmo_violated_dict[gmo.id] = gmo
						gmo_violated_ids.append(gmo.id)

		# NOTE: print out the violation from GMOs
		text_output_path = f"{self.data_writer_cfg['analyzer']}/{self.data_loader_cfg['data_path']}/motorbike_violation.txt"
		with open(text_output_path, 'w') as f:
			for key, value in gmo_violated_dict.items():
				f.write(f"{key}\n"
						f"\t--num::[{value.is_violated_num_people}]"
						f"\t--mov::[{value.is_violated_movement}]"
						f"\t--hel::[{value.is_violated_helmet}]\n")
		for key, value in gmo_violated_dict.items():
			print(f"{key}\n"
				f"--num::[{value.is_violated_num_people}]"
				f"--mov::[{value.is_violated_movement}]"
				f"--hel::[{value.is_violated_helmet}]")

		# DEBUG: draw violation of motorbike
		for index_frame_match, frame, gmos in tqdm(matching_list, desc="Draw violation"):
			# DEBUG: draw bounding boxes, instances in bounding boxes
			image_draw = frame.copy()
			for gmo in gmos:
				# NOTE: draw instance in bounding box
				for instances_in_bbox in gmo.instances_in_bboxes:
					for instance_in_bbox in instances_in_bbox:
						if instance_in_bbox.id[0] != gmo.bboxes_id[-1][0]:  # if not the same with tracker
							break

						image_draw = plot_one_box(
							bbox=instance_in_bbox.bbox,
							img=image_draw,
							color=AppleRGB.values()[instance_in_bbox.class_id],
							label=f"{classes_aic23[instance_in_bbox.class_id]}::{instance_in_bbox.id}"
						)

				# NOTE: draw bounding box
				color_bbox = [AppleRGB.values()[gmo.current_label.train_id + 4]]
				label_bbox = [(f"{classes_aic23[gmo.current_label.train_id]}::{gmo.id % 1000}--"
							   f"{gmo.moving_state}--")]

				if gmo.id in gmo_violated_dict:
					if gmo_violated_dict[gmo.id].is_violated_num_people:
						color_bbox.append((0, 0, 255))
					else:
						color_bbox.append(AppleRGB.values()[gmo.current_label.train_id + 4])
					label_bbox.append(f"num::[{gmo_violated_dict[gmo.id].is_violated_num_people}]--")
					if gmo_violated_dict[gmo.id].is_violated_movement:
						color_bbox.append((0, 0, 255))
					else:
						color_bbox.append(AppleRGB.values()[gmo.current_label.train_id + 4])
					label_bbox.append(f"mov::[{gmo_violated_dict[gmo.id].is_violated_movement}]--")
					if gmo_violated_dict[gmo.id].is_violated_helmet:
						color_bbox.append((0, 0, 255))
					else:
						color_bbox.append(AppleRGB.values()[gmo.current_label.train_id + 4])
					label_bbox.append(f"hel::[{gmo_violated_dict[gmo.id].is_violated_helmet}]--")

				plot_one_box_violated(
					bbox  = gmo.bboxes[-1],
					img   = image_draw,
					colors= color_bbox,
					labels= label_bbox
				)

			cv2.imwrite(os.path.join(folder_output, f"{index_frame_match:04d}.jpg"), image_draw)

	def run_write_draw(self):
		while True:
			(index_frame, frame, data) = self.writer_queue.get()

			if data is None:
				break

	def run(self):
		"""Main run loop."""
		self.run_routine_start()

		# NOTE: detection
		self.run_detector()

		# NOTE: identification
		self.run_identifier()

		# NOTE: tracking and matching
		self.run_tracker_matching()

		# NOTE: analysis
		self.run_analysis()

		self.run_routine_end()

	def run_routine_start(self):
		"""Perform operations when run routine starts. We start the timer."""
		self.start_time = timer()
		if self.verbose:
			cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)

	def run_routine_end(self):
		"""Perform operations when run routine ends."""
		# NOTE: clear detector
		if self.detector is not None:
			self.detector.clear_model_memory()
			self.detector = None

		# NOTE: clear identifier
		if self.identifier is not None:
			self.identifier.clear_model_memory()
			self.identifier = None

		cv2.destroyAllWindows()
		self.stop_time = timer()
		if self.pbar is not None:
			self.pbar.close()

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

	def draw(
			self,
			drawing     : np.ndarray,
			gmos        : list       = None,
			rois        : list       = None,
			mois        : list       = None,
			elapsed_time: float      = None,
	) -> np.ndarray:
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
		# NOTE: Draw ROI
		if rois is not None:
			[roi.draw(drawing=drawing) for roi in self.matcher.rois]
		# NOTE: Draw MOIs
		if mois is not None:
			[moi.draw(drawing=drawing) for moi in self.matcher.mois]
		# NOTE: Draw Vehicles
		if gmos is not None:
			[gmo.draw(drawing=drawing) for gmo in gmos]
		# NOTE: Draw frame index
		# NOTE: Write frame rate
		# fps = self.video_reader.frame_idx / elapsed_time
		# text = f"Frame: {self.video_reader.frame_idx}: {format(elapsed_time, '.3f')}s ({format(fps, '.1f')} fps)"
		# font = cv2.FONT_HERSHEY_SIMPLEX
		# org = (20, 30)
		# NOTE: show the framerate on top left
		# cv2.rectangle(img=drawing, pt1= (10, 0), pt2=(600, 40), color=AppleRGB.BLACK.value, thickness=-1)
		# cv2.putText(img=drawing, text=text, fontFace=font, fontScale=1.0, org=org, color=AppleRGB.WHITE.value, thickness=2)
		return drawing


# MARK - Ultilies


def scaleup_bbox(bbox_xyxy, height_img, width_img, ratio, padding):
	"""Scale up 1.2% or +-40

	Args:
		bbox_xyxy (np.ndarray):
		height_img (int):
		width_img (int):
		ratio (float):
		padding (int):

	Returns:
		bbox_xyxy (np.ndarray):

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


def plot_one_box(bbox, img, color=None, label=None, line_thickness=1):
	"""Plots one bounding box on image img

	Returns:

	"""
	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
	cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
	if label:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

	return img


def plot_one_box_violated(bbox, img, colors=None, labels=None, line_thickness=1):
	"""Plots one bounding box on image img

	Returns:

	"""
	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	colors = colors or [[random.randint(0, 255) for _ in range(3)]]
	c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
	cv2.rectangle(img, c1, c2, colors[0], thickness=tl, lineType=cv2.LINE_AA)
	if labels:
		for index, (color, label) in enumerate(zip(colors, labels)):
			tf = max(tl - 1, 1)  # font thickness
			t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
			c1 = c1[0], c1[1] - (t_size[1] * index)
			c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
			cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
			cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

	return img


def make_directory(folder_path):
	''' Create directory

	Args:
		folder_path (str, Path): the path to directory
	'''
	if not os.path.isdir(folder_path):
		os.makedirs(folder_path)
