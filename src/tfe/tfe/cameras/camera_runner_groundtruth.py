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
"""Ground-truth runner of the web GUI.

``GroundTruthRunner`` runs the counting loop on labelled frames: the boxes and
track ids come from the labelme JSON next to each frame (``group_id`` is the
track id) instead of a detector and a tracker.
"""
import glob
import json
import os
from timeit import default_timer as timer

import cv2
import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from tfe.cameras.camera_runner import CameraRunner
from tfe.cameras.moi import MOI
from tfe.cameras.roi import ROI
from tfe.objects import MovingState
from tfe.objects.instance import Instance
from tfe.tracker.sort.sort_kalman_bbox import KalmanBBoxTrack

# MARK: - GroundTruthRunner

class GroundTruthRunner(CameraRunner):
	"""Ground-truth loop of ``CameraQT6GTH``, without Qt.

	``video_reader`` is the sorted list of frame paths and ``detector`` stays ``None``.
	"""

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
			               f"self.video_reader={self.video_reader is not None}\n"
			               f"{self.result_writer=}\n")
		else:
			logger.info("Camera have been fully configured.")

	def configure_detector(self):
		"""Ground truth needs no detector (``CameraQT6GTH`` skips it, gth:105)."""

	def configure_reader(self):
		"""Configure the video reader."""
		hparams           = self.config.data.copy()

		# Update some parameters for video reader
		hparams.data      = os.path.join(self.config.dirs.data_dir, hparams.dataset, "video", os.path.splitext(hparams.file)[0])
		# self.video_reader = VideoLoader(**hparams)

		# Get list image and json
		self.video_reader = sorted(glob.glob(os.path.join(hparams.data, "*.jpg")) + glob.glob(os.path.join(hparams.data, "*.png")))

	# MARK: Processing

	def _loop(self):
		"""The loop of ``CameraQT6GTH.run()`` (gth:229-332), reading ``self.stop_event``.

		Returns:
			(stopped, None): True if the stop event ended the loop; tqdm wraps the
			frame list and closes itself, so there is no bar to close.
		"""
		stopped = False

		# NOTE: Start timer
		start_time  = timer()
		self.result_writer.start_time = start_time
		self.gmos   = []
		self.is_run = True

		# NOTE: phai them cai nay khong la bi memory leak
		# for images, frame_indexes, files, rel_paths in self.video_reader:
		for img_index, img_path in enumerate(tqdm(self.video_reader, desc=f"{self.config.camera_name}")):
			basename            = os.path.basename(img_path)
			basename_noext, ext = os.path.splitext(basename)
			json_path           = img_path.replace(ext, ".json")
			image               = cv2.imread(img_path)

			if self.stop_event.is_set():
				stopped = True
				break

			# if not self.is_run:  # if the hook running is false, we can out
			# 	break


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

		return stopped, None

	# MARK: Visualize and Debug

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
