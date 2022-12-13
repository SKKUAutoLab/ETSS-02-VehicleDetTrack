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
import threading
import multiprocessing
from queue import Queue
from timeit import default_timer as timer
from typing import Dict
from typing import List

import cv2
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


# MARK: - CameraMultiprocess

class CameraMultiprocess(object):
	"""Camera Multi-thread.

	Attributes:
		config (dict):
			The camera's config.
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			config     : Dict,
			queue_size : int  = 10,
			visualize  : bool = False,
			write_video: bool = False,
			**kwargs
	):
		super().__init__(**kwargs)

		# NOTE: Define attributes
		self.config = config if isinstance(config, Munch) else Munch.fromDict(config)  # A simple check just to make sure
		self.visualize                  = visualize
		self.write_video                = write_video
		self.labels                     = None
		self.rois  : List[ROI]          = None
		self.mois  : List[MOI]          = None
		self.detector                   = None
		self.tracker                    = None
		self.video_reader : VideoReader = None
		self.video_writer : VideoWriter = None
		self.result_writer              = None
		self.pbar                       = None

		# NOTE: Queue
		self.frames_queue     = Queue(maxsize = queue_size)
		self.detections_queue = Queue(maxsize = queue_size)
		self.counting_queue   = Queue(maxsize = queue_size)

		# NOTE: Setup components
		self.configure_labels()
		self.configure_roi()
		self.configure_mois()
		self.configure_detector()
		self.configure_tracker()
		self.configure_gmo()
		self.configure_video_reader()
		self.configure_video_writer()
		self.configure_result_writer()

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
			printw("Camera have not been fully configured. Please check again.")

		if self.visualize:
			cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)

	def configure_labels(self):
		"""Configure the labels."""
		dataset_dir = os.path.join(data_dir, self.config.data.dataset)
		labels = parse_config_from_json(json_path=os.path.join(dataset_dir, "labels.json"))
		labels = Munch.fromDict(labels)
		self.labels = labels.labels

	def configure_roi(self):
		"""Configure the camera's ROIs."""
		hparams   = self.config.roi.copy()
		dataset   = hparams.pop("dataset")
		file      = hparams.pop("file")
		self.rois = ROI.load_rois_from_file(dataset=dataset, file=file, **hparams)

	def configure_mois(self):
		"""Configure the camera's MOIs."""
		hparams   = self.config.moi.copy()
		dataset   = hparams.pop("dataset")
		file      = hparams.pop("file")
		self.mois = MOI.load_mois_from_file(dataset=dataset, file=file, **hparams)

	def configure_detector(self):
		"""Configure the detector."""
		hparams = self.config.detector.copy()
		self.detector = get_detector(labels=self.labels, hparams=hparams)

	def configure_tracker(self):
		"""Configure the tracker."""
		hparams = self.config.tracker.copy()
		self.tracker = get_tracker(hparams=hparams)

	def configure_gmo(self):
		"""Configure the GMO class property."""
		hparams = self.config.gmo.copy()
		GeneralObject.min_travelled_distance = hparams.min_traveled_distance
		GMO.min_traveled_distance            = hparams.min_traveled_distance
		GMO.min_entering_distance            = hparams.min_entering_distance
		GMO.min_hit_streak                   = hparams.min_hit_streak
		GMO.max_age                          = hparams.max_age

	def configure_video_reader(self):
		"""Configure the video reader."""
		hparams = self.config.data.copy()
		self.video_reader = VideoReader(**hparams)

	def configure_video_writer(self):
		"""Configure the video writer."""
		if self.write_video:
			hparams = self.config.output.copy()
			self.video_writer = VideoWriter(output_dir=self.config.dirs.data_output_dir, **hparams)

	def configure_result_writer(self):
		"""Configure the result writer."""
		self.result_writer = AICResultWriter(
			camera_name=self.config.camera_name,
			output_dir=self.config.dirs.data_output_dir
		)

	# MARK: Processing

	def run(self):
		# NOTE: Start timer
		start_time = timer()
		self.result_writer.start_time = start_time

		self.pbar = tqdm(total=self.video_reader.num_frames, desc=f"{self.config.camera_name}")

		# NOTE: Threading for video reader
		process_video_reader = multiprocessing.Process(target=self.run_video_reader)
		process_video_reader.start()

		# NOTE: Threading for detector
		process_detector = multiprocessing.Process(target=self.run_detector)
		process_detector.start()

		# NOTE: Threading for tracker
		process_tracker = multiprocessing.Process(target=self.run_tracker)
		process_tracker.start()

		# NOTE: Threading for result writer
		process_result_writer = multiprocessing.Process(target=self.run_result_writer)
		process_result_writer.start()

		# NOTE: Joins threads when all terminate
		process_video_reader.join()
		process_detector.join()
		process_tracker.join()
		process_result_writer.join()

	def run_video_reader(self):
		for frame_indexes, images in self.video_reader:
			if len(frame_indexes) == 0:
				break
			# NOTE: Push frame index and images to queue
			self.frames_queue.put([frame_indexes, images])

		# NOTE: Push None to queue to act as a stopping condition for next thread
		self.frames_queue.put([None, None])

	def run_detector(self):
		with torch.no_grad():
			while True:
				# NOTE: Get frame indexes and images from queue
				(frame_indexes, images) = self.frames_queue.get()
				if frame_indexes is None:
					break

				# NOTE: Detect (in batch)
				images = padded_resize_image(images=images, size=self.detector.dims[1:3])
				batch_detections = self.detector.detect_objects(frame_indexes=frame_indexes, images=images)

				# NOTE: Associate detections with ROI (in batch)
				for idx, detections in enumerate(batch_detections):
					ROI.associate_detections_to_rois(detections=detections, rois=self.rois)
					batch_detections[idx] = [d for d in detections if d.roi_uuid is not None]

				# NOTE: Push detections to queue
				self.detections_queue.put(batch_detections)

		# NOTE: Push None to queue to act as a stopping condition for next thread
		self.detections_queue.put(None)

	def run_tracker(self):
		while True:
			# NOTE: Get batch detections from queue
			batch_detections = self.detections_queue.get()

			if batch_detections is None:
				break

			for idx, detections in enumerate(batch_detections):
				# NOTE: Track (in batch)
				self.tracker.update(detections=detections)
				gmos = self.tracker.tracks

				# NOTE: Update moving state
				for gmo in gmos:
					gmo.update_moving_state(rois=self.rois)
					gmo.timestamps.append(timer())

				# NOTE: Associate gmos with MOIs
				in_roi_gmos = [o for o in gmos if o.is_confirmed or o.is_counting or o.is_to_be_counted]
				MOI.associate_moving_objects_to_mois(gmos=in_roi_gmos, mois=self.mois, shape_type="polygon")
				to_be_counted_gmos = [o for o in in_roi_gmos if o.is_to_be_counted and o.is_countable is False]
				MOI.associate_moving_objects_to_mois(gmos=to_be_counted_gmos, mois=self.mois, shape_type="linestrip")

				# NOTE: Count
				countable_gmos = [o for o in in_roi_gmos if (o.is_countable and o.is_to_be_counted)]
				for gmo in countable_gmos:
					gmo.moving_state = MovingState.Counted

				# NOTE: Push countable moving objects to queue
				self.counting_queue.put(countable_gmos)

			self.pbar.update(len(batch_detections))

		# NOTE: Push None to queue to act as a stopping condition for next thread
		self.counting_queue.put(None)

	def run_result_writer(self):
		while True:
			# NOTE: Get countable moving objects from queue
			countable_gmos = self.counting_queue.get()
			if countable_gmos is None:
				break

			self.result_writer.write_counting_result(vehicles=countable_gmos)
