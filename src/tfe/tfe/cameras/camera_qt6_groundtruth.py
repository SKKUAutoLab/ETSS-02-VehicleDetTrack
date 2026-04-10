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
import glob
import json
import os
from timeit import default_timer as timer
from typing import Dict, Union
from typing import List

import cv2
import numpy as np
from munch import Munch
from tqdm.auto import tqdm

import torch

from loguru import logger

from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *

from tfe.detectors import get_detector
from tfe.io.video import VideoLoader
from tfe.io.video import VideoWriter
from tfe.objects import GeneralObject
from tfe.objects import GMO
from tfe.objects import MovingState
from tfe.objects.class_label import ClassLabels
from tfe.objects.instance import Instance
from tfe.tracker import get_tracker
from tfe.configuration import data_dir
from tfe.tracker.sort.sort_kalman_bbox import KalmanBBoxTrack
from tfe.utils.aic_result_writer import AICResultWriter
from tfe.utils.config import parse_config_from_json
from tfe.cameras.moi import MOI
from tfe.cameras.roi import ROI


from ultralytics.data.augment import LetterBox

# MARK: - Camera

class CameraQT6GTH(QThread):
	"""Camera QT6 for groundtruth
	"""

	# MARK: Magic Functions

	# signal to send to QT6 GUI
	update_information = pyqtSignal(dict)

	def __init__(
			self,
			config     : Dict,
			visualize  : bool = False,
			write_video: bool = False,
			write_image: bool = False,
			**kwargs
	):
		super().__init__(**kwargs)
		# NOTE: Define attributes
		self.config                    = config if isinstance(config, Munch) else Munch.fromDict(config)  # A simple check just to make sure
		self.visualize                 = visualize
		self.write_video               = write_video
		self.write_image               = write_image
		self.labels                    = None
		self.rois: List[ROI]           = None
		self.mois: List[MOI]           = None
		self.detector                  = None
		self.tracker                   = None
		self.video_reader: VideoLoader = None
		self.video_writer: VideoWriter = None
		self.result_writer             = None
		self.gmos: List[GMO]           = []

		# NOTE: Setup components
		self.configure_labels()
		self.configure_roi()
		self.configure_mois()
		self.configure_detector()
		self.configure_tracker()
		self.configure_gmo()
		self.configure_reader()
		self.configure_result_writer()
		# self.configure_video_writer()

		# NOTE: Final check before running
		self.check_components()

	# MARK: Configure

	def check_components(self):
		"""Check if the camera's components have been successfully defined.
		"""
		if any(c is None for c in (self.rois, self.mois, self.detector, self.tracker, self.video_reader, self.result_writer)):
			logger.warning("Camera have not been fully configured. Please check again.")
			logger.warning(f"\n{self.rois=}\n"
			               f"{self.mois=}\n"
			               f"{self.detector=}\n"
			               f"{self.tracker=}\n"
			               f"{self.video_reader=}\n"
			               f"{self.result_writer=}\n")
		else:
			logger.info("Camera have been fully configured.")

	def configure_labels(self):
		"""Configure the labels."""
		dataset_dir       = os.path.join(data_dir, self.config.data.dataset)
		labels            = parse_config_from_json(json_path = os.path.join(dataset_dir, "labels.json"))
		labels            = Munch.fromDict(labels)
		self.labels       = labels.labels
		self.class_labels = ClassLabels(self.labels)

	def configure_roi(self):
		"""Configure the camera's ROIs."""
		hparams   = self.config.roi.copy()
		dataset   = hparams.pop("dataset")
		file      = hparams.pop("file")
		self.rois = ROI.load_rois_from_file(dataset=dataset, file=file, **hparams)

	def configure_mois(self):
		"""Configure the camera's MOIs."""
		hparams    = self.config.moi.copy()
		dataset    = hparams.pop("dataset")
		file       = hparams.pop("file")
		self.mois  = MOI.load_mois_from_file(dataset=dataset, file=file, **hparams)
		self.mois_display = {}
		for moi in self.mois:
			self.mois_display[moi.uuid] = 0

	def configure_detector(self):
		"""Configure the detector."""
		hparams                 = self.config.detector.copy()
		hparams["class_labels"] = self.class_labels

		# check the path of weights
		if isinstance(hparams.weights, str) and (not os.path.isabs(hparams.weights)):
			hparams["weights"] = os.path.join(self.config.dirs.models_zoo_dir, hparams.weights)
		elif isinstance(hparams.weights, Union[list, tuple]) and len(hparams.weights) > 0 and (not os.path.isabs(hparams.weights[0])):
			hparams["weights"] = [os.path.join(self.config.dirs.models_zoo_dir, w) for w in hparams.weights]
		self.detector = get_detector(hparams=hparams)

	def configure_tracker(self):
		"""Configure the tracker."""
		hparams      = self.config.tracker.copy()
		self.tracker = get_tracker(hparams=hparams)

	def configure_gmo(self):
		"""Configure the GMO class property."""
		hparams                              = self.config.gmo.copy()
		GeneralObject.min_travelled_distance = hparams.min_traveled_distance
		GMO.min_traveled_distance            = hparams.min_traveled_distance
		GMO.min_entering_distance            = hparams.min_entering_distance
		GMO.min_hit_streak                   = hparams.min_hit_streak
		GMO.max_age                          = hparams.max_age

	def configure_reader(self):
		"""Configure the video reader."""
		hparams           = self.config.data.copy()

		# Update some parameters for video reader
		hparams.data      = os.path.join(self.config.dirs.data_dir, hparams.dataset, "video", os.path.splitext(hparams.file)[0])
		# self.video_reader = VideoLoader(**hparams)

		# Get list image and json
		self.video_reader = sorted(glob.glob(os.path.join(hparams.data, "*.jpg")) + glob.glob(os.path.join(hparams.data, "*.png")))

	def configure_video_writer(self):
		"""Configure the video writer."""
		if self.write_video:
			hparams           = self.config.output.copy()

			# Update some parameters for video reader
			hparams.dst = os.path.join(self.config.dirs.data_output_dir, hparams.file)
			self.video_writer = VideoWriter(output_dir=self.config.dirs.data_output_dir, **hparams)

	def configure_result_writer(self):
		"""Configure the result writer."""
		self.result_writer = AICResultWriter(
			camera_name = self.config.camera_name,
			output_dir  = self.config.dirs.data_output_dir
		)

	# MARK: Processing

	def run(self):
		"""The main processing loop.
		"""
		# NOTE: Start timer
		start_time = timer()
		self.result_writer.start_time = start_time
		self.gmos = []

		# NOTE: phai them cai nay khong la bi memory leak
		# for images, frame_indexes, files, rel_paths in self.video_reader:
		for img_index, img_path in enumerate(tqdm(self.video_reader, desc=f"{self.config.camera_name}")):
			basename            = os.path.basename(img_path)
			basename_noext, ext = os.path.splitext(basename)
			json_path           = img_path.replace(ext, ".json")
			image               = cv2.imread(img_path)


			# Load JSON
			instances      = []
			if os.path.exists(json_path):
				with open(json_path, 'r') as f:
					data = json.load(f)

				# Load all detection and add into instance array
				for i, shape in enumerate(data['shapes']):
					confident   = 99.0
					class_id    = self.class_labels.get_id(key="name", value=shape.get('label'))
					class_label = self.class_labels.get_class_label_by_name(shape.get('label'))
					bbox_points = np.array(shape.get('points'))
					bbox_xyxy   = np.array([
							int(bbox_points[:, 0].min()),
							int(bbox_points[:, 1].min()),
							int(bbox_points[:, 0].max()),
							int(bbox_points[:, 1].max()),
						]).astype(np.int32)
					instances.append(
						Instance(
							frame_index = img_index,
							bbox        = bbox_xyxy,
							confidence  = confident,
							class_label = class_label,
							label       = class_label,
							class_id    = class_id,
							track_id    = int(shape.get('group_id')),
							image_size  = image.shape[:2]  # (H, W)
						)
					)

			# NOTE: Associate detections with ROI (in batch)
			ROI.associate_detections_to_rois(instances=instances, rois=self.rois)
			instances = [d for d in instances if d.roi_uuid is not None]

			# First we need add 1 age to all tracklet, so if they get the new instance, they will be minus age
			for idx, trk in enumerate(self.gmos):
				self.gmos[idx].time_since_update +=1

			# NOTE: Find which Instance below to which tracklet (self.gmos)
			to_add = []
			for instance in instances:
				for idx, gmo in enumerate(self.gmos):
					if instance.track_id == gmo.id:
						self.gmos[idx].update_gmo(instance)
						self.gmos[idx].time_since_update = max(0, self.gmos[idx].time_since_update - 1)
						to_add.append(instance)
						break
			instances = [instance for instance in instances if instance not in to_add]

			# NOTE: If instance not belong to any tracklet (self.gmos), create the new one
			for instance in instances:
				tracklet        = KalmanBBoxTrack.track_from_detection(instance)
				tracklet.id     = instance.track_id
				self.gmos.append(tracklet)

			# NOTE: Remove gmos that are out of ROI or death track
			self.gmos = [trk for trk in self.gmos if trk.time_since_update <= self.tracker.max_age]

			# NOTE: Update moving state
			for gmo in self.gmos:
				gmo.update_moving_state(rois=self.rois)
				gmo.timestamps.append(timer())

			# NOTE: Associate gmos with MOI
			in_roi_gmos = [o for o in self.gmos if o.is_confirmed or o.is_counting or o.is_to_be_counted]
			MOI.associate_moving_objects_to_mois(gmos=in_roi_gmos, mois=self.mois, shape_type="polygon")
			to_be_counted_gmos = [o for o in in_roi_gmos if o.is_to_be_counted and o.is_countable is False]
			MOI.associate_moving_objects_to_mois(gmos=to_be_counted_gmos, mois=self.mois, shape_type="linestrip")

			# NOTE: Count
			countable_gmos = [o for o in in_roi_gmos if (o.is_countable and o.is_to_be_counted)]
			self.result_stored_counting = self.extract_mois_information(countable_gmos)  # NOTE: store for GUI

			self.result_writer.write_counting_result(vehicles=countable_gmos)

			# NOTE: Update moving state
			for gmo in countable_gmos:
				gmo.moving_state = MovingState.Counted

			# NOTE: Visualize and Debug
			elapsed_time = timer() - start_time
			self.post_process(image=image, elapsed_time=elapsed_time)

		# NOTE: need to delete because the output txt has to be finish becafore evaluation,
		# we run Qthread so the thread might be run in different time
		del self.result_writer

		# send the stop process
		self.update_information.emit({
			"frame_index" : None,
			"result_count": None,
			"result_frame": None,
			"is_run"      : False
		}) # NOTE: send to GUI

	def stop(self):
		"""Sets run flag to False and waits for thread to finish"""
		# self.run_flag = False
		self.wait()

	# MARK: Visualize and Debug

	def extract_mois_information(self, countable_gmos):
		for vehicle in countable_gmos:
			moi_id = vehicle.moi_uuid
			self.mois_display[moi_id] += 1
		results = []
		for key, values in self.mois_display.items():
			results.append(f"Movement {key} : {values}")
		return results

	def post_process(self, image: np.ndarray, elapsed_time: float):
		"""Post processing step.
		"""
		# NOTE: Visualize results
		if not self.visualize and not self.write_video:
			return
		result = self.draw(drawing=image, elapsed_time=elapsed_time)
		if self.visualize:
			# Display the resulting frame
			self.result_stored_image_drawed = result  # NOTE: store for GUI
			self.update_information.emit({
				"frame_index"  : None,
				"result_count" : self.result_stored_counting,
				"result_frame" : self.result_stored_image_drawed,
				"is_run"       : True
			}) # NOTE: send to GUI
			# Frame per second show on display
			self.msleep(3)
		if self.write_video:
			if self.video_writer is None:
				self.config.output.shape = result.shape # Same as input (H, W, C)
				self.configure_video_writer()

			self.video_writer.write_frame(image=result)

	def draw(self, drawing: np.ndarray, elapsed_time: float):
		"""Visualize the results on the drawing.
		"""
		# NOTE: Draw ROI
		[roi.draw(drawing=drawing) for roi in self.rois]
		# NOTE: Draw MOIs
		[moi.draw(drawing=drawing) for moi in self.mois]
		# NOTE: Draw Vehicles
		[gmo.draw(drawing=drawing) for gmo in self.gmos]
		# NOTE: Draw frame index
		# NOTE: Write frame rate
		# fps  = self.video_reader.index / elapsed_time
		# text = f"Frame: {self.video_reader.index}: {format(elapsed_time, '.3f')}s ({format(fps, '.1f')} fps)"
		# font = cv2.FONT_HERSHEY_SIMPLEX
		# org  = (20, 30)
		# NOTE: show the framerate on top left
		# cv2.rectangle(img=drawing, pt1= (10, 0), pt2=(600, 40), color=AppleRGB.BLACK.value, thickness=-1)
		# cv2.putText(img=drawing, text=text, fontFace=font, fontScale=1.0, org=org, color=AppleRGB.WHITE.value, thickness=2)
		return drawing


