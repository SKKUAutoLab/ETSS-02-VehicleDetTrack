# ==================================================================== #
# File name: camera.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/27/2021
# ==================================================================== #
import os
from timeit import default_timer as timer
from typing import Dict
from typing import List

import cv2
import numpy as np
from munch import Munch
from tqdm.auto import tqdm

import torch

from tfe.detector import get_detector
from tfe.io import AICResultWriter
from tfe.io import VideoReader
from tfe.io import VideoWriter
from tfe.ops import AppleRGB, padded_resize_image
from tfe.road_objects import GeneralObject
from tfe.road_objects import GMO
from tfe.road_objects import MovingState
from tfe.tracker import get_tracker
from tfe.utils import data_dir
from tfe.utils import parse_config_from_json
from tfe.utils import printw
from .moi import MOI
from .roi import ROI


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
			**kwargs
	):
		super().__init__(**kwargs)
		# TODO: Define attributes
		self.config                    = config if isinstance(config, Munch) else Munch.fromDict(config)  # A simple check just to make sure
		self.visualize                 = visualize
		self.write_video               = write_video
		self.labels                    = None
		self.rois: List[ROI]           = None
		self.mois: List[MOI]           = None
		self.detector                  = None
		self.tracker                   = None
		self.video_reader: VideoReader = None
		self.video_writer: VideoWriter = None
		self.result_writer             = None
		self.gmos: List[GMO]           = []

		# TODO: Setup components
		self.configure_labels()
		self.configure_roi()
		self.configure_mois()
		self.configure_detector()
		self.configure_tracker()
		self.configure_gmo()
		self.configure_video_reader()
		self.configure_video_writer()
		self.configure_result_writer()

		# TODO: Final check before running
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
			printw("Camera have not been fully configured. Please check again.")
			printw(f"{self.rois=}\n"
				   f"{self.mois=}\n"
				   f"{self.detector=}\n"
				   f"{self.tracker=}\n"
				   f"{self.video_reader=}\n"
				   f"{self.result_writer=}\n")

		if self.visualize:
			cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)

	def configure_labels(self):
		"""Configure the labels."""
		dataset_dir = os.path.join(data_dir           , self.config.data.dataset)
		labels      = parse_config_from_json(json_path = os.path.join(dataset_dir, "labels.json"))
		labels      = Munch.fromDict(labels)
		self.labels = labels.labels

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
		hparams       = self.config.detector.copy()
		self.detector = get_detector(labels=self.labels, hparams=hparams)

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

	def configure_video_reader(self):
		"""Configure the video reader."""
		hparams           = self.config.data.copy()
		self.video_reader = VideoReader(**hparams)

	def configure_video_writer(self):
		"""Configure the video writer."""
		if self.write_video:
			hparams           = self.config.output.copy()
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
		# TODO: Start timer
		start_time = timer()
		self.result_writer.start_time = start_time

		# TODO: Loop through all frames in self.video_reader
		pbar = tqdm(total=self.video_reader.num_frames, desc=f"{self.config.camera_name}")

		# NOTE: phai them cai nay khong la bi memory leak
		with torch.no_grad():
			for frame_indexes, images in self.video_reader:
				if len(frame_indexes) == 0:
					break

				# TODO: Detect (in batch)
				images = padded_resize_image(images=images, size=self.detector.dims[1:3])
				batch_detections = self.detector.detect_objects(frame_indexes=frame_indexes, images=images)

				# TODO: Associate detections with ROI (in batch)
				for idx, detections in enumerate(batch_detections):
					ROI.associate_detections_to_rois(detections=detections, rois=self.rois)
					batch_detections[idx] = [d for d in detections if d.roi_uuid is not None]

				# TODO: Track (in batch)
				for idx, detections in enumerate(batch_detections):
					self.tracker.update(detections=detections)
					self.gmos = self.tracker.tracks

					# TODO: Update moving state
					for gmo in self.gmos:
						gmo.update_moving_state(rois=self.rois)
						gmo.timestamps.append(timer())

					# TODO: Associate gmos with MOI
					in_roi_gmos = [o for o in self.gmos if o.is_confirmed or o.is_counting or o.is_to_be_counted]
					MOI.associate_moving_objects_to_mois(gmos=in_roi_gmos, mois=self.mois, shape_type="polygon")
					to_be_counted_gmos = [o for o in in_roi_gmos if o.is_to_be_counted and o.is_countable is False]
					MOI.associate_moving_objects_to_mois(gmos=to_be_counted_gmos, mois=self.mois, shape_type="linestrip")

					# TODO: Count
					countable_gmos = [o for o in in_roi_gmos if (o.is_countable and o.is_to_be_counted)]
					self.result_writer.write_counting_result(vehicles=countable_gmos)
					for gmo in countable_gmos:
						gmo.moving_state = MovingState.Counted

					# TODO: Visualize and Debug
					elapsed_time = timer() - start_time
					self.post_process(image=images[idx], elapsed_time=elapsed_time)

				pbar.update(len(frame_indexes))  # Update pbar

		# TODO: Finish
		pbar.close()
		cv2.destroyAllWindows()

	# MARK: Visualize and Debug

	def post_process(self, image: np.ndarray, elapsed_time: float):
		"""Post processing step.
		"""
		# TODO: Visualize results
		if not self.visualize and not self.write_video:
			return
		result = self.draw(drawing=image, elapsed_time=elapsed_time)
		if self.visualize:
			cv2.imshow("image", result)
			cv2.waitKey(1)
		if self.write_video:
			self.video_writer.write_frame(image=result)

	def draw(self, drawing: np.ndarray, elapsed_time: float):
		"""Visualize the results on the drawing.
		"""
		# TODO: Draw ROI
		[roi.draw(drawing=drawing) for roi in self.rois]
		# TODO: Draw MOIs
		[moi.draw(drawing=drawing) for moi in self.mois]
		# TODO: Draw Vehicles
		[gmo.draw(drawing=drawing) for gmo in self.gmos]
		# TODO: Draw frame index
		# NOTE: Write frame rate
		fps  = self.video_reader.frame_idx / elapsed_time
		text = f"Frame: {self.video_reader.frame_idx}: {format(elapsed_time, '.3f')}s ({format(fps, '.1f')} fps)"
		font = cv2.FONT_HERSHEY_SIMPLEX
		org  = (20, 30)
		# NOTE: show the framerate on top left
		# cv2.rectangle(img=drawing, pt1= (10, 0), pt2=(600, 40), color=AppleRGB.BLACK.value, thickness=-1)
		# cv2.putText(img=drawing, text=text, fontFace=font, fontScale=1.0, org=org, color=AppleRGB.WHITE.value, thickness=2)
		return drawing
