import os
import sys
import argparse
import json
from dataclasses import dataclass

import numpy as np
from loguru import logger
from tqdm import tqdm

import cv2
from shapely.geometry import Polygon, box

classes = [
	"unidentified",
	"others",
	"pedestrian",
	"micromobility",
	"car",
	"bus",
	"small truck",
	"truck",
]

@dataclass
class BBox:
	x_min     : float = 0.0
	y_min     : float = 0.0
	x_max     : float = 0.0
	y_max     : float = 0.0
	confidence: float = 0.0
	class_id  : int   = 0
	class_name: str   = "unidentified"
	track_id  : int   = 0
	frame_id  : int   = 0
	img_width : int   = 0
	img_height: int   = 0

	def __repr__(self):
		return f"BBox(class_name={self.class_name}, track_id={self.track_id}, frame_id={self.frame_id}, x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max})"

def filename_last_element_key(filename):
	# remove extension, split on '_' and take the last element
	base = os.path.splitext(filename)[0]
	last = base.split('_')[-1].split('-')[-1]
	try:
		return int(last)
	except ValueError:
		return last


def load_rmois(json_rmois):
	if not os.path.exists(json_rmois):
		return None

	with open(json_rmois, "r") as f:
		return json.load(f)


def load_image_folder(args):
	if not os.path.exists(args.input_image_folder):
		logger.error(f"Image folder {args.input_image_folder} does not exist.")
		return None

	image_files = sorted(
		[
			f
			for f in os.listdir(args.input_image_folder)
			if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
		],
		key=filename_last_element_key
	)
	return image_files


def load_video(args):
	cap = cv2.VideoCapture(args.input_video)
	if not cap.isOpened():
		logger.error(f"Error opening video file {args.input_video}")
		return None
	return cap


def check_valid_bbox_in_roi(roi, bbox):
	roi_polygon =Polygon(roi)
	bbox_poly = box(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
	return roi_polygon.intersects(bbox_poly)


def load_and_filter_tracking_results(args):
	# create output folder if it does not exist
	output_tracking_folder = args.output_tracking_folder
	output_images     = os.path.join(output_tracking_folder, "images")
	output_images_roi = os.path.join(output_tracking_folder, "images_roi")
	os.makedirs(output_images, exist_ok=True)
	os.makedirs(output_images_roi, exist_ok=True)
	output_label_file = os.path.join(output_tracking_folder, f"labels_result.txt")

	# Load rmois
	rmois = load_rmois(args.json_rmois)
	roi   = rmois["roi"][0]["points"]

	# Load and Sort list of files in the tracking result folder
	if not os.path.exists(args.input_tracking_folder):
		logger.error(f"Tracking result folder {args.input_tracking_folder} does not exist.")
		return
	list_of_files = sorted(os.listdir(args.input_tracking_folder), key=filename_last_element_key)

	# Load video and image folder
	if args.input_video is not None:
		video_capture = load_video(args)
		if video_capture is None:
			return
		img_width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		img_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	if args.input_image_folder is not None:
		image_files = load_image_folder(args)
		if image_files is None:
			return
		# Load the first image to get width and height
		first_image = cv2.imread(os.path.join(args.input_image_folder, image_files[0]))
		img_height, img_width = first_image.shape[:2]

	# Check if the number of tracking result files matches the number of images
	logger.info(f"Number of tracking result files: {len(list_of_files)}")
	logger.info(f"Number of images in folder: {len(image_files) if args.input_image_folder is not None else 'N/A'}")
	if len(list_of_files) != len(image_files):
		logger.warning("The number of tracking result files does not match the number of images.")
		return

	# Load tracking results from the specified folder
	num_bbox          = 0
	num_bbox_filtered = 0
	pbar = tqdm(total=len(list_of_files), desc=f"Processing tracking results {os.path.basename(output_tracking_folder)}")
	with open(output_label_file, "w") as f_out:
		for index_image, (image_file, label_file) in enumerate(zip(image_files, list_of_files)):
			img = cv2.imread(os.path.join(args.input_image_folder, image_file))

			# create label output file for this image
			label_content = {
				"version"    : "2.5.0",
				"flags"      : {},
				"shapes"     : [],
				"imagePath"  : image_file,
				"imageData"  : None,
				"imageHeight": img_height,
				"imageWidth" : img_width
			}

			# Draw ROI on the image copy
			img_roi = img.copy()
			pts = np.array(roi, np.int32)
			pts = pts.reshape((-1, 1, 2))
			cv2.polylines(img_roi, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

			# Load tracking result file
			label_path = os.path.join(args.input_tracking_folder, label_file)
			with open(label_path, "r") as f:
				lines = f.readlines()
				for line in lines:
					# <class_id x_center_normal y_center_normal w_normal h_normal confident_score track_id>
					# 4 0.62176114320755 0.6264861822128296 0.04770026355981827 0.06884710490703583 0.9499133825302124 1
					items = line.replace("\n","").replace("\r","").split(" ")
					num_bbox += 1

					bbox_tracking = BBox()
					bbox_tracking.x_min      = (float(items[1]) * img_width) - ((float(items[3]) * img_width) / 2)
					bbox_tracking.y_min      = (float(items[2]) * img_height) - ((float(items[4]) * img_height) / 2)
					bbox_tracking.x_max      = (float(items[1]) * img_width) + ((float(items[3]) * img_width) / 2)
					bbox_tracking.y_max      = (float(items[2]) * img_height) + ((float(items[4]) * img_height) / 2)
					bbox_tracking.class_id   = int(items[0])
					bbox_tracking.class_name = classes[bbox_tracking.class_id] if bbox_tracking.class_id < len(classes) else "unidentified"
					bbox_tracking.confidence = float(items[5])
					bbox_tracking.track_id  = int(items[6])
					bbox_tracking.frame_id   = index_image

					if check_valid_bbox_in_roi(roi, bbox_tracking):
						# frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility, <skipped>
						f_out.write(f"{bbox_tracking.frame_id},"
						            f"{bbox_tracking.track_id},"
						            f"{int(bbox_tracking.x_min)},"
						            f"{int(bbox_tracking.y_min)},"
						            f"{int(bbox_tracking.x_max - bbox_tracking.x_min)},"
						            f"{int(bbox_tracking.y_max - bbox_tracking.y_min)},"
						            f"1,"
						            f"{bbox_tracking.class_id},"
						            f"1"
						            f"\n")
						label_content["shapes"].append({
							"label": bbox_tracking.class_name,
							"description": None,
							"points": [
								[
									int(bbox_tracking.x_min),
									int(bbox_tracking.y_min)
								],
								[
									int(bbox_tracking.x_max),
									int(bbox_tracking.y_min)
								],
								[
									int(bbox_tracking.x_max),
									int(bbox_tracking.y_max)
								],
								[
									int(bbox_tracking.x_min),
									int(bbox_tracking.y_max)
								]
							],
							"group_id": bbox_tracking.track_id,
							"difficult": False,
							"direction": 0,
							"shape_type": "rectangle",
							"flags": {}
						})
						num_bbox_filtered += 1

			# Save images and images with roi overlay
			cv2.imwrite(os.path.join(output_images, image_file), img)
			cv2.imwrite(os.path.join(output_images_roi, image_file), img_roi)

			# Save label content as json
			json.dump(label_content, open(os.path.join(output_images_roi, f"{os.path.splitext(image_file)[0]}.json"), "w"), indent=2)

			pbar.update(1)
	pbar.close()

	# DEBUG: Print the filtered tracking results
	logger.info(f"Total number of bounding boxes before filtering: {num_bbox}")
	logger.info(f"Total number of bounding boxes after filtering: {num_bbox_filtered}")

	if args.input_video is not None:
		video_capture.release()
	cv2.destroyAllWindows()

def main():
	parser = argparse.ArgumentParser(description="Config parser for running")
	parser.add_argument(
		"--input_video",
		help="The input video."
	)
	parser.add_argument(
		"--input_image_folder",
		help="The input image folder."
	)
	parser.add_argument(
		"--json_rmois",
		default="SUWON_cctv.json",
		help="The input json file containing rmois."
	)
	parser.add_argument(
		"--input_tracking_folder",
		default="/SUWON_cctv/tracking_result",
		help="The input tracking result folder."
	)
	parser.add_argument(
		"--output_tracking_folder",
		default="/SUWON_cctv/tracking_label",
		help="The input tracking result folder."
	)

	# Parse the arguments
	args = parser.parse_args()

	# load video or image folder
	if args.input_video is None and args.input_image_folder is None:
		logger.error("Either --input_video or --input_image_folder must be provided.")
		return


	# Load and filter tracking results
	load_and_filter_tracking_results(args)


if __name__ == "__main__":
	main()



