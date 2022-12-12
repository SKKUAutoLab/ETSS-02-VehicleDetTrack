# ==================================================================== #
# File name: camera_non_detection.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 09/27/2022
# ==================================================================== #
import os
import sys
import glob
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
from tfe.detector import Detection
from .moi import MOI
from .roi import ROI


# MARK: - Camera Non Detection

class CameraNonDetection(object):
	"""Camera Non Detection
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

	def load_all_detection(self, folder_path):
		video_detections = []

		label_conversion = {
			'10' : Munch({ "name": "car"   , "id": 1, "train_id": 0, "category": "vehicle", "catId": 1, "color": [0, 0, 142] }),  # pedestrian :: car
			'4'  : Munch({ "name": "people", "id": 2, "train_id": 1, "category": "vehicle", "catId": 1, "color": [0, 0, 70 ] })   # vehicle    :: truck
		}

		size_ori = [1280.0, 720.0]
		size_new = [768.0, 448.0]

		# Load txt
		list_txts     = glob.glob(os.path.join(folder_path, "*.txt"))
		def order_name(elem):
			return int(os.path.splitext(os.path.basename(elem))[0])
		list_txts.sort(key=order_name)

		frame_indexes = 0
		for txt_path in tqdm(list_txts):
			detections = []
			idx        = 0
			with open(txt_path, "r") as f_read:
				lines = f_read.readlines()
				for line in lines:
					words = line.replace("\n", "").replace("\r", "").split(" ")

					bbox_xyxy = np.array([
						int(float(words[1]) * size_new[0] / size_ori[0] ),
						int(float(words[2]) * size_new[1] / size_ori[1] ),
						int(float(words[3]) * size_new[0] / size_ori[0] ),
						int(float(words[4]) * size_new[1] / size_ori[1] )
					], np.int32)

					# Wrong size bounding box
					if bbox_xyxy[0] == bbox_xyxy[2] or \
							bbox_xyxy[1] == bbox_xyxy[3]:
						continue

					detections.append(
						Detection(
							frame_index = frame_indexes + idx,
							bbox        = bbox_xyxy,
							confidence  = 1.0,
							label       = label_conversion[words[0]]
						)
					)
					idx = idx + 1
			video_detections.append(detections)
			frame_indexes = frame_indexes + 1

		return video_detections

	def run(self):
		"""The main processing loop.
		"""
		# TODO: Start timer
		start_time = timer()
		self.result_writer.start_time = start_time

		# load detection from txts
		video_detections  = self.load_all_detection(f"/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/RnT-TFE/data/carla/bbox/{self.config['camera_name']}/")
		index_batch_frame = 0

		# TODO: Loop through all frames in self.video_reader
		pbar = tqdm(total=self.video_reader.num_frames, desc=f"{self.config.camera_name}")

		# NOTE: phai them cai nay khong la bi memory leak, out of memory, GPU memory
		with torch.no_grad():
			for frame_indexes, images in self.video_reader:

				if len(frame_indexes) == 0:
					break

				# TODO: Detect (in batch)
				images = padded_resize_image(images=images, size=self.detector.dims[1:3])
				# batch_detections = self.detector.detect_objects(frame_indexes=frame_indexes, images=images)
				batch_detections = []
				for index_frame, detections in enumerate(video_detections):
					if index_batch_frame <= index_frame < index_batch_frame + len(frame_indexes):
						batch_detections.append(detections)
				index_batch_frame = index_batch_frame + len(frame_indexes)

				# DEBUG:
				# print(batch_detections)
				# print(len(batch_detections))
				# print(batch_detections.shape)
				# print(images[0].shape)
				# print(self.detector.dims)
				# print(self.video_reader.dims)
				# sys.exit()

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
		fps  = self.video_reader.frame_idx / elapsed_time
		text = f"Frame: {self.video_reader.frame_idx}: {format(elapsed_time, '.3f')}s ({format(fps, '.1f')} fps)"
		font = cv2.FONT_HERSHEY_SIMPLEX
		org  = (20, 30)
		cv2.rectangle(img=drawing, pt1= (10, 0), pt2=(600, 40), color=AppleRGB.BLACK.value, thickness=-1)
		cv2.putText(img=drawing, text=text, fontFace=font, fontScale=1.0, org=org, color=AppleRGB.WHITE.value, thickness=2)
		return drawing
