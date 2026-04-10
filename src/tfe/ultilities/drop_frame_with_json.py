# python
import argparse
import os
import random
import shutil
from typing import List

from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def filename_last_element_key(filename):
	# remove extension, split on '_' and take the last element
	base = os.path.splitext(filename)[0]
	last = base.split('_')[-1].split('-')[-1]
	try:
		return int(last)
	except ValueError:
		return last


def get_list_images(folder: str, exts: set = IMAGE_EXTS) -> List[str]:
	items = sorted([
		f for f in os.listdir(folder)
		if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in exts],
		key=filename_last_element_key
	)
	return items


def main():
	# Init variable
	folder_input  = "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_labels/23_SUWON/images_roi"
	folder_output = "/media/sugarubuntu/DataSKKU4/4_Dataset/vlc_record/Korea_cctv/tracking_labels/23_SUWON/images_roi_dropframe"
	skip_frame    = 5

	os.makedirs(folder_output, exist_ok=True)
	list_img = get_list_images(folder_input)

	for index, img_name in enumerate(tqdm(list_img)):
		base_name, ext = os.path.splitext(img_name)
		json_name = f"{base_name}.json"

		# check exist file
		input_img_path  =  os.path.join(folder_input, img_name)
		input_json_path =  os.path.join(folder_input, json_name)
		if not os.path.exists(input_img_path) or not os.path.exists(input_json_path):
			print(f"File not found: {input_img_path} or {input_json_path}")
			continue

		# Decide to copy or skip
		if index % skip_frame == 0 or index == len(list_img) - 1:
			# Copy image file
			output_img_path  = os.path.join(folder_output, img_name)
			shutil.copy(input_img_path, output_img_path)

			# Copy json file
			output_json_path = os.path.join(folder_output, json_name)
			shutil.copy(input_json_path, output_json_path)

if __name__ == "__main__":
	main()