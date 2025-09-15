import os
import glob
import sys

import csv
import numpy as np
import math
import pandas as pd

from tqdm import tqdm


def convert_xlsx_to_csv(xlsx_path, csv_path):
	import openpyxl
	wb = openpyxl.load_workbook(xlsx_path)
	sheet_names = wb.sheetnames
	df = pd.read_excel(xlsx_path, sheet_name=sheet_names[0])  # sheet_name is optional
	df.to_csv(csv_path, index=False)  # index=False prevents pandas from writing a row index to the CSV.


def convert_xlsxes():
	# Init parameter
	folder_in = "/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/tfe/utils/eval_count/gt/"

	# Load list file
	xlsx_list = glob.glob(os.path.join(folder_in, "*.xlsx"))	

	for xlsx_path in tqdm(xlsx_list):
		convert_xlsx_to_csv(xlsx_path, xlsx_path.replace("xlsx","csv"))


def check_csv(csv_path, csv_path_out, video_info):
	with open(csv_path_out, 'w', newline='') as csvfile_out:
		with open(csv_path, newline='') as csvfile:
			lines = csvfile.readlines()
			for line in lines:
				words = line.replace("\n","").split(",")
				if words[1].isnumeric():
					# Get video_id from name
					for index_info, (video_name, info) in enumerate(video_info.items()):
						if words[0] == video_name:
							words[0] = index_info + 1  # because array start from 0

					# Get class_id
					if words[3] == "car" or words[3] == "car ":
						words[3] = 0
					elif words[3] == "truck" or words[3] == "truch":
						words[3] = 1

					# write
					for index_word, word in enumerate(words):
						csvfile_out.write(str(word))
						if index_word != len(words) - 1:
							csvfile_out.write(",")
						else:
							csvfile_out.write("\n")


def check_csvs():
	# Init parameter
	folder_in = "/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/tfe/utils/eval_count/gt/"
	video_info = {
		"cam_1"     : {"frame_num": 3000 , "movement_num": 4},
		"cam_1_dawn": {"frame_num": 3000 , "movement_num": 4},
		"cam_1_rain": {"frame_num": 2961 , "movement_num": 4},
		"cam_2"     : {"frame_num": 18000, "movement_num": 4},
		"cam_2_rain": {"frame_num": 3000 , "movement_num": 4},
		"cam_3"     : {"frame_num": 18000, "movement_num": 4},
		"cam_3_rain": {"frame_num": 3000 , "movement_num": 4},
		"cam_4"     : {"frame_num": 27000, "movement_num": 12},
		"cam_4_dawn": {"frame_num": 4500 , "movement_num": 12},
		"cam_4_rain": {"frame_num": 3000 , "movement_num": 12},
		"cam_5"     : {"frame_num": 18000, "movement_num": 12},
		"cam_5_dawn": {"frame_num": 3000 , "movement_num": 12},
		"cam_5_rain": {"frame_num": 3000 , "movement_num": 12},
		"cam_6"     : {"frame_num": 18000, "movement_num": 12},
		"cam_6_snow": {"frame_num": 3000 , "movement_num": 12},
		"cam_7"     : {"frame_num": 14400, "movement_num": 12},
		"cam_7_dawn": {"frame_num": 2400 , "movement_num": 12},
		"cam_7_rain": {"frame_num": 3000 , "movement_num": 12},
		"cam_8"     : {"frame_num": 3000 , "movement_num": 6},
		"cam_9"     : {"frame_num": 3000 , "movement_num": 12},
		"cam_10"    : {"frame_num": 2111 , "movement_num": 3},
		"cam_11"    : {"frame_num": 2111 , "movement_num": 3},
		"cam_12"    : {"frame_num": 1997 , "movement_num": 3},
		"cam_13"    : {"frame_num": 1966 , "movement_num": 3},
		"cam_14"    : {"frame_num": 3000 , "movement_num": 2},
		"cam_15"    : {"frame_num": 3000 , "movement_num": 2},
		"cam_16"    : {"frame_num": 3000 , "movement_num": 2},
		"cam_17"    : {"frame_num": 3000 , "movement_num": 2},
		"cam_18"    : {"frame_num": 3000 , "movement_num": 2},
		"cam_19"    : {"frame_num": 3000 , "movement_num": 2},
		"cam_20"    : {"frame_num": 3000 , "movement_num": 2}
	}

	# Load list file
	csvs_list = glob.glob(os.path.join(folder_in, "*.csv"))

	for csv_path in tqdm(csvs_list):
		check_csv(csv_path, csv_path.replace("_raw.csv",".csv"), video_info)


if __name__ == "__main__":
	# convert_xlsxes()
	check_csvs()
