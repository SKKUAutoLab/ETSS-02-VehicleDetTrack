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

from tfe.detectors import get_detector

from tfe.io.video import VideoLoader
from tfe.io.video import VideoWriter
from tfe.objects import GeneralObject
from tfe.objects import GMO
from tfe.objects import MovingState
from tfe.objects.class_label import ClassLabels
from tfe.tracker import get_tracker
from tfe.configuration import data_dir
from tfe.utils.aic_result_writer import AICResultWriter
from tfe.utils.config import parse_config_from_json
from tfe.cameras.moi import MOI
from tfe.cameras.roi import ROI


from ultralytics.data.augment import LetterBox

# MARK: - Camera

class Camera(object):
	"""Camera
	"""

	# MARK: Magic Functions

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
		# self.configure_writer()

		# NOTE: Final check before running
		self.check_components()

	# MARK: Configure

	def check_components(self):
		"""Check if the camera's components have been successfully defined.
		"""
		if self.rois is None or \
				self.mois is None or \
				self.detector is None or \
				self.tracker is None or \
				self.video_reader is None or \
				self.result_writer is None:
			logger.warning("Camera have not been fully configured. Please check again.")
			logger.warning(f"\n{self.rois=}\n"
						   f"{self.mois=}\n"
						   f"{self.detector=}\n"
						   f"{self.tracker=}\n"
						   f"{self.video_reader=}\n"
						   f"{self.result_writer=}\n")

		if self.visualize:
			cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)

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
		hparams      = self.config.data.copy()

		# Update some parameters for video reader
		hparams.data = os.path.join(self.config.dirs.data_dir, hparams.dataset, "video", hparams.file)
		self.video_reader = VideoLoader(**hparams)

	def configure_writer(self):
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

		# configure letterbox for image preprocessing (used in YOLO models)
		letterbox = LetterBox(
			new_shape=(
				self.config.detector.shape[1],
				self.config.detector.shape[2]
			),  # [C, H, W] in config.roi.shape  Target size
			auto       = False, # Use exact new_shape
			scale_fill = False, # Maintain aspect ratio with padding
			scaleup    = True,  # Allow upscaling if needed
			stride     = 32,    # Model stride (common for YOLO)
			center     = False
		)

		# NOTE: Loop through all frames in self.video_reader
		pbar = tqdm(total=self.video_reader.num_frames, desc=f"{self.config.camera_name}")

		# NOTE: phai them cai nay khong la bi memory leak
		with torch.no_grad():
			for images, frame_indexes, files, rel_paths  in self.video_reader:
				if len(frame_indexes) == 0:
					break

				# NOTE: Detect (in batch)
				# images = padded_resize_image(images=images, size=self.detector.dims[1:3])
				# batch_instances = self.detector.detect_objects(frame_indexes=frame_indexes, images=images)
				images = [letterbox(image = im) for im in images]
				batch_instances = self.detector.detect(
					indexes=frame_indexes, images=images
				)

				# DEBUG:
				# print("*" * 100)
				# for attr_name, attr_value in self.video_writer.__dict__.items():
				# 	if not attr_name.startswith('__'):  # Filter out special attributes
				# 		print(f"{attr_name}: {attr_value}")
				# print(f"{result.shape=}")
				# print("*" * 100)
				# exit()
				# print("*" * 100)
				# for idx, instances in enumerate(batch_instances):
				# 	for instance in instances:
				# 		print(f"{instance.bbox}")
				# print("*" * 100)

				# NOTE: Associate detections with ROI (in batch)
				for idx, instances in enumerate(batch_instances):
					ROI.associate_detections_to_rois(instances=instances, rois=self.rois)
					batch_instances[idx] = [d for d in instances if d.roi_uuid is not None]

				# NOTE: Track (in batch)
				for idx, instances in enumerate(batch_instances):
					self.tracker.update(instances=instances)
					self.gmos = self.tracker.tracks

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
					self.result_writer.write_counting_result(vehicles=countable_gmos)
					for gmo in countable_gmos:
						gmo.moving_state = MovingState.Counted

					# NOTE: Visualize and Debug
					elapsed_time = timer() - start_time
					self.post_process(image=images[idx], elapsed_time=elapsed_time)

				pbar.update(len(frame_indexes))  # Update pbar

		# NOTE: Finish
		pbar.close()
		cv2.destroyAllWindows()

	# MARK: Visualize and Debug

	def post_process(self, image: np.ndarray, elapsed_time: float):
		"""Post processing step.
		"""
		# NOTE: Visualize results
		if not self.visualize and not self.write_video:
			return
		result = self.draw(drawing=image, elapsed_time=elapsed_time)
		if self.visualize:
			cv2.imshow("image", result)
			cv2.waitKey(1)
		if self.write_video:
			if self.video_writer is None:
				self.config.output.shape = result.shape # Same as input (H, W, C)
				self.configure_writer()

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
		fps  = self.video_reader.index / elapsed_time
		text = f"Frame: {self.video_reader.index}: {format(elapsed_time, '.3f')}s ({format(fps, '.1f')} fps)"
		font = cv2.FONT_HERSHEY_SIMPLEX
		org  = (20, 30)
		# NOTE: show the framerate on top left
		# cv2.rectangle(img=drawing, pt1= (10, 0), pt2=(600, 40), color=AppleRGB.BLACK.value, thickness=-1)
		# cv2.putText(img=drawing, text=text, fontFace=font, fontScale=1.0, org=org, color=AppleRGB.WHITE.value, thickness=2)
		return drawing
