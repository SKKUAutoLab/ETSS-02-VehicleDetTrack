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
"""Detection runner of the web GUI.

``CameraRunner`` runs the detection loop on the web GUI's job thread: every
drawn frame goes to ``RunnerCallbacks.on_info``, and a ``threading.Event``
stops the run. ``configure_*`` and ``draw`` are inherited from ``Camera``.
"""
from threading import Event
from time import sleep
from timeit import default_timer as timer
from typing import Optional

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from tfe.cameras.camera import Camera
from tfe.cameras.moi import MOI
from tfe.cameras.roi import ROI
from tfe.objects import MovingState
from tfe.utils.runner_callbacks import RunnerCallbacks

# MARK: - CameraRunner

class CameraRunner(Camera):
	"""Detection, tracking and counting loop of ``CameraQT6``, without Qt.
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			config      : dict,
			visualize   : bool                      = False,
			write_video : bool                      = False,
			write_image : bool                      = False,
			*,
			stop_event  : Optional[Event]           = None,
			callbacks   : Optional[RunnerCallbacks] = None,
			emit_sleep_s: float                     = 0.0,
	):
		self._init_runtime(stop_event=stop_event, callbacks=callbacks, emit_sleep_s=emit_sleep_s)
		Camera.__init__(self, config, visualize, write_video, write_image)

	def _init_runtime(
			self,
			stop_event  : Optional[Event]           = None,
			callbacks   : Optional[RunnerCallbacks] = None,
			emit_sleep_s: float                     = 0.0,
	):
		"""Set the run-time state the loop needs besides the camera's components."""
		self.stop_event   = stop_event if stop_event is not None else Event()
		self.callbacks    = callbacks if callbacks is not None else RunnerCallbacks()
		self.emit_sleep_s = emit_sleep_s
		self.is_run       = False

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

	# MARK: Adjust Running

	def request_stop(self):
		"""Ask the loop to stop; it checks the event once per batch."""
		self.stop_event.set()

	# MARK: Processing

	def run(self):
		"""Run the loop, close the result file, send the final dict, then report the end.

		Same order as ``CameraQT6.run()`` (qt6:278-291). An exception goes to
		``callbacks.on_error`` and is re-raised.
		"""
		try:
			stopped, pbar = self._loop()

			# NOTE: the writer thread encodes the queued frames inside the timed run
			if self.video_writer is not None:
				self.video_writer.close()

			# NOTE: need to delete because the output txt has to be finish becafore evaluation,
			# we run Qthread so the thread might be run in different time
			self.result_writer.close()

			# send the stop process
			self.callbacks.on_info({
				"frame_index" : None,
				"result_count": None,
				"result_frame": None,
				"is_run"      : False
			})  # NOTE: send to GUI

			# NOTE: Finish
			if pbar is not None:
				pbar.close()
		except Exception as exc:
			self.callbacks.on_error(exc)
			raise
		self.callbacks.on_finished(stopped)

	def _loop(self):
		"""The loop of ``CameraQT6.run()`` (qt6:216-276), reading ``self.stop_event``.

		The event is never cleared here (qt6:214 cleared it): every run gets its own.

		Returns:
			(stopped, pbar): True if the stop event ended the loop, and the open tqdm bar.
		"""
		stopped = False

		# NOTE: Start timer
		start_time  = timer()
		self.result_writer.start_time = start_time
		self.is_run = True

		# NOTE: Loop through all frames in self.video_reader
		pbar = tqdm(total=self.video_reader.num_frames, desc=f"{self.config.camera_name}")

		# NOTE: phai them cai nay khong la bi memory leak
		with torch.no_grad():
			for images, frame_indexes, files, rel_paths in self.video_reader:
				if len(frame_indexes) == 0:  # which mean no frame, so there is no index
					break

				if self.stop_event.is_set():
					stopped = True
					break

				# if not self.is_run:  # if the hook running is false, we can out
				# 	break

				# NOTE: Detect (in batch)
				batch_instances = self.detector.detect(
					indexes=frame_indexes, images=images
				)

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
					self.result_stored_counting = self.extract_mois_information(countable_gmos)  # NOTE: store for GUI

					self.result_writer.write_counting_result(vehicles=countable_gmos)

					# NOTE: Update moving state
					for gmo in countable_gmos:
						gmo.moving_state = MovingState.Counted

					# NOTE: Visualize and Debug
					elapsed_time = timer() - start_time
					self.post_process(image=images[idx], elapsed_time=elapsed_time)

				pbar.update(len(frame_indexes))  # Update pbar

		return stopped, pbar

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

		``CameraQT6.post_process`` (qt6:311-334) with ``callbacks.on_info`` for the
		Qt signal and ``sleep(emit_sleep_s)`` for ``msleep(3)`` (0 s by default since
		2026-10-29: Open Q12 revised). It never opens a window, unlike ``Camera.post_process``.
		"""
		# NOTE: Visualize results
		if not self.visualize and not self.write_video:
			return
		result = self.draw(drawing=image, elapsed_time=elapsed_time)
		if self.visualize:
			# Display the resulting frame
			self.result_stored_image_drawed = result  # NOTE: store for GUI
			self.callbacks.on_info({
				"frame_index"  : None,
				"result_count" : self.result_stored_counting,
				"result_frame" : self.result_stored_image_drawed,
				"is_run"       : True
			}) # NOTE: send to GUI
			# Frame per second show on display
			sleep(self.emit_sleep_s)
		if self.write_video:
			if self.video_writer is None:
				self.config.output.shape = result.shape # Same as input (H, W, C)
				self.configure_video_writer()

			self.video_writer.write_frame(image=result)
