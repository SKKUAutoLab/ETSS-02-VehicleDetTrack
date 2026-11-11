#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Paths and scores of a GUI run, with the formulas, console lines and file lines of the removed Qt GUI.

``derive_run_paths`` is ``set_video_information`` (view:212-225), ``score_one`` is
``evaluate_one_result`` (view:483-520) and ``score_all`` is ``evaluate_all_result``
(view:522-545): same formulas, console lines and file lines. This module must not
import torch: the web layer imports it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from tfe.utils.eval_nwrmse import evaluate_one_video

__all__ = [
	"BASE_FACTOR",
	"SEGMENT_PERIOD",
	"VIDEO_INFO",
	"RunPaths",
	"derive_run_paths",
	"score_all",
	"score_one",
]


# MARK: - Constants (view:110-125; tests/web/test_gui_evaluation.py compares them with the view)

VIDEO_INFO = {
	"23_SUWON": {
		"frame_num" : 6001, "movement_num": 10, "index_from": 0, "index_to": 6000, "number_of_classes": 8,
		"video_path": "data/Korea_cctv/video/23_SUWON.mp4"
	},
	"34_SUWON": {
		"frame_num" : 6001, "movement_num": 11 , "index_from": 0, "index_to": 6000, "number_of_classes": 8,
		"video_path": "data/Korea_cctv/video/34_SUWON.mp4"
	},
	"40_SUWON": {
		"frame_num" : 6001, "movement_num": 15 , "index_from": 0, "index_to": 6000, "number_of_classes": 8,
		"video_path": "data/Korea_cctv/video/40_SUWON.mp4"
	},
	"45a_SUWON": {
		"frame_num" : 6001, "movement_num": 13 , "index_from": 0, "index_to": 6000, "number_of_classes": 8,
		"video_path": "data/Korea_cctv/video/45a_SUWON.mp4"
	},
	"60_SUWON": {
		"frame_num" : 6001, "movement_num": 6 , "index_from": 0, "index_to": 6000, "number_of_classes": 8,
		"video_path": "data/Korea_cctv/video/60_SUWON.mp4"
	},
	"12_01_SUWON": {
		"frame_num" : 6001, "movement_num": 5 , "index_from": 0, "index_to": 6000, "number_of_classes": 8,
		"video_path": "data/Korea_cctv/video/12_01_SUWON.mp4"
	},
	"12_02_SUWON": {
		"frame_num" : 6001, "movement_num": 5 , "index_from": 0, "index_to": 6000, "number_of_classes": 8,
		"video_path": "data/Korea_cctv/video/12_02_SUWON.mp4"
	},
}

SEGMENT_PERIOD = 10
BASE_FACTOR    = 5.514405  # 5.514405 :: 1.014405


# MARK: - Paths

@dataclass(frozen=True)
class RunPaths:
	"""What ``set_video_information`` derives from a selected video file or frame folder."""
	dataset       : str
	video_name    : str
	config        : str
	output        : str
	output_run_all: str
	groundtruth   : str
	video_label   : str


def derive_run_paths(video_path: str) -> RunPaths:
	"""``<root>/<dataset>/video/<name>[.ext]`` → the paths of view:213-218 and the label of view:225."""
	basename_noext = os.path.splitext(os.path.basename(video_path))[0]
	dataset_dir    = os.path.dirname(os.path.dirname(video_path))
	dataset        = os.path.basename(dataset_dir)
	return RunPaths(
		dataset        = dataset,
		video_name     = basename_noext,
		config         = os.path.join(dataset_dir, "configs", f"{basename_noext}.yaml"),
		output         = os.path.join(dataset_dir, "temps", f"{basename_noext}.txt"),
		output_run_all = os.path.join(dataset_dir, "outputs", f"{dataset}.txt"),
		groundtruth    = os.path.join(dataset_dir, "groundtruths", f"{basename_noext}.txt"),
		video_label    = f"Dataset : {dataset} -- Video: {basename_noext}",
	)


# MARK: - Scores

def score_one(video_name: str, groundtruth: str, output: str, period_s: float, video_index: int) -> dict:
	"""Score one run, print it and append 3 lines to ``output``, as ``evaluate_one_result`` does.

	Args:
		video_index (int):
			Position in ``VIDEO_INFO`` whose ``frame_num`` enters the efficiency;
			-1 (the last entry) for a single run, as in the Qt GUI.
	"""
	wRMSE, vehicle_num = evaluate_one_video(video_name, VIDEO_INFO, groundtruth, output, SEGMENT_PERIOD)

	total_process_time = period_s
	video_total_frame  = list(VIDEO_INFO.values())[video_index]["frame_num"]
	score_effetiveness = wRMSE / vehicle_num
	score_efficiency   = 1 - (total_process_time * BASE_FACTOR) / (1.1 * float(video_total_frame))
	score_s1           = 0.3 * score_efficiency + 0.7 * score_effetiveness

	print(f"vehicle_num  : {vehicle_num}\n"
		  f"wRMSE\t\t : {wRMSE:.6f}\n"
		  f"nwRMSE\t     : {wRMSE / vehicle_num:.6f}\n"
		  f"period\t     : {total_process_time:.2f}\n"
		  f"index_video  : {video_index}\n"
		  f"effetiveness : {score_effetiveness:.6f}\n"
		  f"efficiency   : {score_efficiency:.6f}\n"
		  f"score_s1\t : {score_s1:.6f}\n"
		#   f"base_factor\t : {BASE_FACTOR:.6f}\n"
		  )

	with open(output, "a") as f_append:
		f_append.write(f"effetiveness : {score_effetiveness:.2f}\n"
					   f"efficiency   : {score_efficiency:.2f}\n"
					   f"score_s1\t : {score_s1:.2f}\n"
					   )

	return {
		"video"        : video_name,
		"vehicle_num"  : vehicle_num,
		"wRMSE"        : wRMSE,
		"period_s"     : total_process_time,
		"effectiveness": score_effetiveness,
		"efficiency"   : score_efficiency,
		"score_s1"     : score_s1,
	}


def score_all(per_video: list, output_run_all: str) -> dict:
	"""Score a completed RUN ALL, print it and rewrite ``output_run_all``, as ``evaluate_all_result`` does."""
	wRMSE_array          = [result["wRMSE"] for result in per_video]
	vehicleNum_array     = [result["vehicle_num"] for result in per_video]
	period_process_array = [result["period_s"] for result in per_video]

	video_total_frame  = sum([value["frame_num"] for value in VIDEO_INFO.values()])
	total_process_time = sum(period_process_array)

	score_effetiveness = sum(wRMSE_array) / sum(vehicleNum_array)
	score_efficiency   = 1 - (total_process_time * BASE_FACTOR) / (1.1 * float(video_total_frame))
	score_s1           = 0.3 * score_efficiency + 0.7 * score_effetiveness

	# NOTE: the view prints with ``{self.x=}``, so its lines start with the expression text
	print(f"self.wRMSE_array={wRMSE_array!r} -- self.vehicleNum_array={vehicleNum_array!r} -- {period_process_array}")
	print(f"sum(self.wRMSE_array)={sum(wRMSE_array)!r} -- sum(self.vehicleNum_array)={sum(vehicleNum_array)!r} -- sum(self.period_process_array)={sum(period_process_array)!r}")
	print(f"score_effetiveness={score_effetiveness!r}")
	print(f"score_efficiency={score_efficiency!r}")
	print(f"score_s1={score_s1!r}")
	# print(f"base_factor={BASE_FACTOR!r}")
	print("*********************")
	print("FINISHED")
	print("*********************")

	with open(output_run_all, "w") as f_write:
		f_write.write(f"effetiveness : {score_effetiveness:.2f}\n"
					  f"efficiency   : {score_efficiency:.2f}\n"
					  f"score_s1\t : {score_s1:.2f}\n"
					  )

	return {
		"vehicle_num"  : sum(vehicleNum_array),
		"wRMSE"        : sum(wRMSE_array),
		"period_s"     : total_process_time,
		"effectiveness": score_effetiveness,
		"efficiency"   : score_efficiency,
		"score_s1"     : score_s1,
	}
