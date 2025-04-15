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

from tqdm import tqdm

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
			'area'          : 0.0,
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
	# file_path_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/gt.txt"
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


def create_json_coco_format_for_result():
	print("Create json file from result of AIC")

	# NOTE: init
	file_path_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/outputs_s2_v8_det_v8_iden/final_result_s2.txt"
	file_path_ou = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/re_aic2023_train.json"
	# file_path_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/output_aic24/final_result.txt"
	# file_path_ou = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/re_aic2024_train.json"
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


def coco_evaluation():
	# NOTE: init
	# annFile = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/instances_val2014.json"
	# annFile = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/instances_val2014_test.json"
	# resFile = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/instances_val2014_fakebbox100_results.json"
	annFile = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/gt_aic2023_train.json"
	resFile = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/re_aic2023_train.json"
	# annFile = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/gt_aic2024_train.json"
	# resFile = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/re_aic2024_train.json"

	annType = ['segm', 'bbox', 'keypoints']
	annType = annType[1]  # specify type here
	prefix  = 'person_keypoints' if annType == 'keypoints' else 'instances'
	print('Running demo for *%s* results.' % (annType))

	# NOTE: load files
	cocoGt = COCO(annFile)
	cocoDt = cocoGt.loadRes(resFile)

	imgIds = sorted(cocoGt.getImgIds())

	# DEBUG: only run 100 images
	# imgIds = imgIds[0:100]
	# imgId  = imgIds[np.random.randint(100)]

	# NOTE: running evaluation
	cocoEval = COCOeval(cocoGt, cocoDt, annType)
	cocoEval.params.imgIds = imgIds
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()

# endregion

# NOTE: TEST ------------------------------------------------------------------


def load_json():
	annFile_in = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/instances_val2014.json"
	annFile_ou = "/media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/instances_val2014_test.json"
	data_json = json.load(open(annFile_in))

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

	with open(annFile_ou, 'w') as f:
		json.dump(data_json, f)


def main():
	# load_json()

	# create json file from txt groundtruth
	create_json_coco_format_for_groundtruth()

	# create json file from txt result
	create_json_coco_format_for_result()

	# Evaluation
	coco_evaluation()

	pass

if __name__ == "__main__":
	main()
