import argparse
import os
import sys
import glob
import json

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
# import skimage.io as io
import pylab

import copy

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument(
	"--anno_file_json_path",
	default="/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/gt_aic2023_train.json",
	help="Path to annotation json file."
)
parser.add_argument(
	"--result_file_json_path",
	default="/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/re_aic2023_train.json",
	help="Path to result json file."
)
parser.add_argument(
	"--result_file_txt_path",
	default="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/outputs_s2_v8_det_v8_iden/final_result_s2.txt",
	help="Path to result txt file."
)
parser.add_argument(
	"--result_evaluation_txt_path",
	default="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/outputs_s2_v8_det_v8_iden/final_result_s2_evaluation.txt",
	help="Path to result of evaluation file."
)


# NOTE: CREATE COCO FORMAT DATASET----------------------------------------------------

# region CREATE COCO FORMAT DATASET


def create_categories_aic2023():
	categories = [
		{'supercategory': 'vehicle', 'id': 1, 'name': 'motorbike'},
		{'supercategory': 'person' , 'id': 2, 'name': 'DHelmet'},
		{'supercategory': 'person' , 'id': 3, 'name': 'DNoHelmet'},
		{'supercategory': 'person' , 'id': 4, 'name': 'P1Helmet' },
		{'supercategory': 'person' , 'id': 5, 'name': 'P1NoHelmet'},
		{'supercategory': 'person' , 'id': 6, 'name': 'P2Helmet' },
		{'supercategory': 'person' , 'id': 7, 'name': 'P2NoHelmet'}
	]
	return categories


def create_categories_aic2024():
	categories = [
		{'supercategory': 'vehicle', 'id': 1, 'name': 'motorbike'},
		{'supercategory': 'person' , 'id': 2, 'name': 'DHelmet'},
		{'supercategory': 'person' , 'id': 3, 'name': 'DNoHelmet'},
		{'supercategory': 'person' , 'id': 4, 'name': 'P1Helmet' },
		{'supercategory': 'person' , 'id': 5, 'name': 'P1NoHelmet'},
		{'supercategory': 'person' , 'id': 6, 'name': 'P2Helmet' },
		{'supercategory': 'person' , 'id': 7, 'name': 'P2NoHelmet'},
		{'supercategory': 'person',  'id': 8, 'name': 'P0Helmet'},
		{'supercategory': 'person',  'id': 9, 'name': 'P0NoHelmet'}
	]
	return categories


def create_images(labels):
	# {'license': 1, 'file_name': 'COCO_val2014_000000560744.jpg', 'height': 480, 'width': 640, 'id': 560744}
	# gt: <video_id>, <frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <class>
	images = []
	# for label in labels:
	# we need the add more image than annotation, or video have
	for video_index in range(1,101):
		for image_index in range(1,220):
			filename_notext = f"{video_index:03d}{image_index:05d}"
			filename        = f"{filename_notext}.jpg"
			image_id        = int(filename_notext)
			images.append({
				'license'  : 1,
				'file_name': filename,
				'height'   : 1080,
				'width'    : 1920,
				'id'       : image_id
			})
	return images


def create_annotations_2024(labels):
	# {'segmentation': [[239.97, 260.24, 222.04, 270.49]],
	# 'area': 2765.1486,
	# 'iscrowd': 0,
	# 'image_id': 558840,
	# 'bbox': [199.84, 200.46, 77.71, 70.88],
	# 'category_id': 58,
	# 'id': 156
	# }
	# gt_2024: <video_id>, <frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <class>
	images = []
	object_index = 0
	for label in labels:
		object_index += 1
		image_id = int(f"{int(label[0]):03d}{int(label[1]):05d}")
		images.append({
			'segmentation'  : [],
			'area'          : 0.0,
			'iscrowd'       : 0,
			'image_id'      : image_id,
			'bbox'          : [
				float(label[2]),
				float(label[3]),
				float(label[4]),
				float(label[5])
			],
			'category_id'   : int(label[6]),
			'id'            : object_index
		})
	return images


def create_annotations_2023(labels):
	# {'segmentation': [[239.97, 260.24, 222.04, 270.49]],
	# 'area': 2765.1486,
	# 'iscrowd': 0,
	# 'image_id': 558840,
	# 'bbox': [199.84, 200.46, 77.71, 70.88],
	# 'category_id': 58,
	# 'id': 156
	# }
	# gt_2023: <video_id>, <frame>, <track_id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <class>
	images = []
	object_index = 0
	for label in labels:
		object_index += 1
		image_id = int(f"{int(label[0]):03d}{int(label[1]):05d}")
		images.append({
			'segmentation'  : [],
			'area'          : float(label[5]) * float(label[6]),
			'iscrowd'       : 0,
			'image_id'      : image_id,
			'bbox'          : [
				float(label[3]),
				float(label[4]),
				float(label[5]),
				float(label[6])
			],
			'category_id'   : int(label[7]),
			'id'            : object_index
		})
	return images


def create_json_coco_format_for_groundtruth():
	"""Create json file from groundtruth of AIC
		info        = {}
		images      = ...
		licenses    = {}
		annotations = ...
		categories  = ...
	"""
	print("Create json file from groundtruth of AIC")

	# NOTE: init
	file_path_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/gt.txt"
	file_path_ou = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/gt_aic2023_train.json"
	# file_path_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/gt_clean.txt"
	# file_path_ou = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/gt_aic2024_train.json"
	data_json    = {
		"info"     : {},
		"licenses" : {}
	}

	# NOTE: load label
	labels = []
	with open(file_path_in, 'r') as f_open:
		# gt_2024: <video_id>, <frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <class>
		# gt_2023: <video_id>, <frame>, <track_id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <class>
		lines = f_open.readlines()
		for line in lines:
			labels.append([float(word) for word in line.replace('\n', '').split(',')])

	# NOTE: add cagecories
	data_json["categories"]  = create_categories_aic2023()
	data_json["images"]      = create_images(labels)
	data_json["annotations"] = create_annotations_2023(labels)

	# data_json["categories"]  = create_categories_aic2024()
	# data_json["images"]      = create_images(labels)
	# data_json["annotations"] = create_annotations_2024(labels)

	# NOTE: write out json
	with open(file_path_ou, 'w') as f:
		json.dump(data_json, f)


def create_json_coco_format_for_result(args):
	print("Create json file from result of AIC")

	# NOTE: init
	# file_path_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/outputs_s2_v8_det_v8_iden/final_result_s2.txt"
	# file_path_ou = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/re_aic2023_train.json"
	# file_path_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/output_aic24/final_result.txt"
	# file_path_ou = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/re_aic2024_train.json"
	file_path_in = args.result_file_txt_path
	file_path_ou = args.result_file_json_path
	data_json = []

	# NOTE: load label
	labels = []
	with open(file_path_in, 'r') as f_open:
		# result: <video_id>, <frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <class>, <confidence>
		lines = f_open.readlines()
		for line in lines:
			labels.append([float(word) for word in line.replace('\n', '').split(',')])

	# NOTE: assign label in json
	for label in labels:
		filename_notext = f"{int(label[0]):03d}{int(label[1]):05d}"
		id              = int(filename_notext)
		data_json.append({
			"image_id"   : id,
			"category_id": int(label[6]),
			"bbox"       : [
				float(label[2]),
				float(label[3]),
				float(label[4]),
				float(label[5])
			],
			"score"      : float(label[7])
		})

	# NOTE: write out json
	with open(file_path_ou, 'w') as f:
		json.dump(data_json, f)

# endregion

# NOTE: COCO EVALUATOR --------------------------------------------------------

# region COCO EVALUATOR


def coco_evaluation(args):
	# NOTE: init
	# anno_file_path = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/instances_val2014.json"
	# anno_file_path = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/instances_val2014_test.json"
	# resu_file_path = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/instances_val2014_fakebbox100_results.json"
	# anno_file_path = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/gt_aic2023_train.json"
	# resu_file_path = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/re_aic2023_train.json"
	# anno_file_path = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/gt_aic2024_train.json"
	# resu_file_path = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/re_aic2024_train.json"
	anno_file_path = args.anno_file_json_path
	resu_file_path = args.result_file_json_path

	annType = ['segm', 'bbox', 'keypoints']
	annType = annType[1]  # specify type here
	prefix  = 'person_keypoints' if annType == 'keypoints' else 'instances'
	print('Running demo for *%s* results.' % (annType))

	# NOTE: load files
	cocoGt = COCO(anno_file_path)
	cocoDt = cocoGt.loadRes(resu_file_path)

	imgIds = sorted(cocoGt.getImgIds())

	# DEBUG: only run 100 images
	# imgIds = imgIds[0:100]
	# imgId  = imgIds[np.random.randint(100)]

	# get list of IDs
	catIds = copy.deepcopy(cocoGt.getCatIds())

	# clean the write file
	with open(args.result_evaluation_txt_path, 'w') as f:
		f.write("")

	# NOTE: running evaluation each subclass
	cocoEval = COCOeval(cocoGt, cocoDt, annType)
	for catId in catIds:
		print("\n******")
		print(f"class_id: {catId}")
		print("******")
		cocoEval.params.catIds = [catId]  # id of class
		cocoEval.params.imgIds = imgIds
		cocoEval.evaluate()
		cocoEval.accumulate()
		cocoEval.summarize()

		with open(args.result_evaluation_txt_path, 'a') as f:
			sys.stdout = f
			print("\n******")
			print(f"class_id: {catId}")
			print("******")
			cocoEval.summarize()
			sys.stdout = sys.__stdout__

	# NOTE: running evaluation overall
	print("\n******")
	print("Overall")
	print("******")
	cocoEval.params.catIds = catIds  # id of class
	cocoEval.params.imgIds = imgIds
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()

	with open(args.result_evaluation_txt_path, 'a') as f:
		sys.stdout = f
		print("\n******")
		print("Overall")
		print("******")
		cocoEval.summarize()
		sys.stdout = sys.__stdout__


# endregion

# NOTE: TEST ------------------------------------------------------------------


def load_json():
	anno_file_path_in = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/instances_val2014.json"
	anno_file_path_ou = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/instances_val2014_test.json"
	data_json = json.load(open(anno_file_path_in))

	for key, value in data_json.items():
		print(key)

	data_json["info"]     = {}
	data_json["licenses"] = {}

	for image in tqdm(data_json["images"]):
		image.pop('flickr_url', None)
		image.pop('coco_url', None)
		image.pop('date_captured', None)

	# print(data_json["images"])

	for annotation in tqdm(data_json["annotations"]):
		annotation["segmentation"] = []
		annotation["area"]         = 0.0

	# print(data_json["annotations"][0])

	with open(anno_file_path_ou, 'w') as f:
		json.dump(data_json, f)


def main():
	args = parser.parse_args()

	# load_json()

	# create json file from txt groundtruth
	# create_json_coco_format_for_groundtruth()

	# create json file from txt result
	create_json_coco_format_for_result(args)

	# Evaluation
	coco_evaluation(args)

	pass


if __name__ == "__main__":
	main()
