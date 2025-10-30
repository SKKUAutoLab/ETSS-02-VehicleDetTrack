
import numpy as np
import math
import os
from collections import OrderedDict


def parse_txt(txt_path, parse_array):
	'''From the our code'''
	# <gen_time> <video_id> <frame_id> <movement_id> <vehicle_class_id> 1:car--2:people
	with open(txt_path, 'r') as fp:
		lines = fp.readlines()
		for line in lines:
			words            = line.rstrip("\n").split(" ")
			frame_id         = int(words[2]) - 1
			movement_id      = int(words[3]) - 1
			vehicle_class_id = int(words[4]) - 1
			parse_array[frame_id, movement_id, vehicle_class_id] = 1


def compute_nwRMSE(segment_period, predict_array, groundtruth_array):
	# weight
	wVect = np.asarray(np.arange(1, segment_period + 1)) / (segment_period * (segment_period + 1) / 2.0)
	frame_num, movement_num, vehicle_class_num = predict_array.shape
	lst            = range(0, frame_num)
	interval       = int(math.ceil(frame_num / float(segment_period)))
	segment_lists  = [lst[i : i + interval] for i in range(0, len(lst), interval)]
	groundtruth_count_array = np.zeros(movement_num)
	prediction_count_array  = np.zeros(movement_num)
	nwRMSE_array            = np.zeros((movement_num, 2))
	wRMSE_array             = np.zeros((movement_num, 2))
	vehicleNum_array        = np.zeros((movement_num, 2))

	for movement_id in range(0, movement_num):
		groundtruth_count_array[movement_id] = np.sum(groundtruth_array[:, movement_id, :])
		prediction_count_array[movement_id]  = np.sum(predict_array[:, movement_id, :])
		for tId in range(0, 2):
			# wRMSE
			diffVectCul = np.zeros(segment_period)
			for segment_id, frames in enumerate(segment_lists):
				diff = np.square(sum(predict_array[0:frames[-1], movement_id, tId]) - sum(groundtruth_array[0:frames[-1], movement_id, tId]))
				diffVectCul[segment_id] = diff
			wRMSE = np.sqrt(np.dot(wVect, diffVectCul))

			# num
			vehicle_num = np.sum(groundtruth_array[:, movement_id, tId])
			vehicleNum_array[movement_id, tId] = vehicle_num

			# for print only
			if vehicle_num == 0:
				wRMSE_array[movement_id, tId] = 0
			else:
				wRMSE_array[movement_id, tId] = wRMSE / vehicle_num

			#nwRMSE
			if wRMSE > vehicle_num:
				nwRMSE = 0
			else:
				if vehicle_num == 0:
					nwRMSE = 0
				else:
					nwRMSE = 1 - wRMSE / vehicle_num
			nwRMSE_array[movement_id, tId] = nwRMSE

	print("")
	print_string = " moveID: "
	separate_line = " --------"
	for moveId, val in enumerate(np.sum(wRMSE_array, axis=1).tolist()):
		print_string += f"{int(moveId+1):04d} | "
		separate_line += "-------"
	print(print_string)
	print(separate_line)

	print_string = " gt cnt: "
	separate_line = " --------"
	for val in groundtruth_count_array.tolist():
		print_string += f"{int(val):04d} | "
		separate_line += "-------"
	print(print_string)
	print(separate_line)

	print_string = " pd cnt: "
	separate_line = " --------"
	for val in prediction_count_array.tolist():
		print_string += f"{int(val):04d} | "
		separate_line += "-------"
	print(print_string)
	print(separate_line)

	print_string = " nwRMSE: "
	for moveId, val in enumerate(np.sum(nwRMSE_array, axis=1).tolist()):
		print_string += f"{val:2.2f} | "
	print(print_string)

	wRMSE_array = np.multiply(nwRMSE_array, vehicleNum_array)
	return np.sum(wRMSE_array), np.sum(vehicleNum_array)


def evaluate_one_video(video_name, video_info, groundtruth_path, prediction_path, segment_period):
	# get video information
	movement_num = video_info[video_name]["movement_num"]
	frame_num    = video_info[video_name]["frame_num"]
	index_from   = video_info[video_name]["index_from"]
	index_to     = video_info[video_name]["index_to"]

	if index_to > frame_num:
		index_to = frame_num

	# parse groundtruth
	groundtruth_array = np.zeros((frame_num, movement_num, 2))
	if not os.path.exists(groundtruth_path):
		raise f"Do not have groundtruth {groundtruth_path}"
	parse_txt(groundtruth_path, groundtruth_array)

	# parse prediction
	prediction_array = np.zeros((frame_num, movement_num, 2))
	if not os.path.exists(prediction_path):
		raise f"Do not have prediction {prediction_path}"
	parse_txt(prediction_path, prediction_array)

	# extract the period of time in frame
	groundtruth_array = groundtruth_array[index_from:index_to, :, :]
	prediction_array = prediction_array[index_from:index_to, :, :]

	# compute nwRMSE
	wRMSE, vehicle_num = compute_nwRMSE(segment_period, prediction_array, groundtruth_array)
	return wRMSE, vehicle_num


def main():
	# Load hyperparameter
	groundtruth_folder = "/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/data/carla_weather/groundtruths"
	prediction_folder  = "/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/data/carla_weather/outputs"
	# prediction_folder  = groundtruth_folder
	video_info = {
		"Town10HD_location_2": {"frame_num": 2000 , "movement_num": 12, "index_from": 0, "index_to": 2000},
		"Town10HD_location_4": {"frame_num": 2000 , "movement_num": 6 , "index_from": 0, "index_to": 2000},
		"Town10HD_location_6": {"frame_num": 1761 , "movement_num": 4 , "index_from": 0, "index_to": 1761},
		"Town10HD_location_7": {"frame_num": 2000 , "movement_num": 2 , "index_from": 0, "index_to": 2000}
	}
	segment_period = 10  # segment number for nwRMSE, (second)

	# Init parameter
	video_num     = len(video_info.keys())
	wRMSEVec      = np.zeros(video_num)
	vehicleNumVec = np.zeros(video_num)

	# evaluate each video
	for video_id, (video_name, info) in enumerate(video_info.items()):
		# init path
		groundtruth_path = os.path.join(groundtruth_folder, f"{video_name}.txt")
		prediction_path  = os.path.join(prediction_folder, f"{video_name}.txt")

		# DEBUG:
		print(f"\n\n{groundtruth_path=}\n{prediction_path=}")

		# compute nwRMSE
		wRMSE, vehicle_num = evaluate_one_video(
			video_name,
			video_info,
			groundtruth_path,
			prediction_path,
			segment_period
		)
		wRMSEVec[video_id]      = wRMSE
		vehicleNumVec[video_id] = vehicle_num
		print(f"{video_name} \n"
		      f"vehicle_num : {vehicle_num}\n"
		      f"wRMSE       : {wRMSE:.6f}\n"
		      f"nwRMSE      : {wRMSE / vehicle_num:.6f}\n"
		      )

	print(f"{sum(wRMSEVec)=} -- {sum(vehicleNumVec)=}")
	score_effetiveness = sum(wRMSEVec) / sum(vehicleNumVec)

	base_factor       = 1.014405
	video_total_frame = sum([value["frame_num"] for value in video_info.values()])
	time = 480
	score_efficiency = 1 - (time * base_factor) / (1.1 * float(video_total_frame))

	score_f1 = 0.3 * score_efficiency + 0.7 * score_effetiveness
	print(f"\ns1: {score_f1:.6f}; effectiveness: {score_effetiveness:.6f}; efficiency: {score_efficiency:.6f}")


if __name__ == "__main__":
	main()
