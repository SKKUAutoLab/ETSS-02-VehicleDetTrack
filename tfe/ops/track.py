# ==================================================================== #
# File name: track.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/29/2021
# ==================================================================== #
from math import asin
from math import cos
from math import pow
from math import sin
from math import sqrt
from typing import Callable

import numpy as np

from tfe.utils import printe


# MARK: - Calculate Distance

def get_distance_function(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
	if name in ["chebyshev", "chebyshev_distance", "chebyshev_dist", "chebyshev_func", "chebyshev_function"]:
		return chebyshev_distance
	elif name in ["cosine", "cosine_distance", "cosine_func", "cosine_func"]:
		return cosine_distance
	elif name in ["euclidean", "euclidean_distance", "euclidean_func", "euclidean_func"]:
		return euclidean_distance
	elif name in ["haversine", "haversine_distance", "haversine_func", "haversine_func"]:
		return haversine_distance
	elif name in ["hausdorff", "hausdorff_distance", "hausdorff_func", "hausdorff_func"]:
		return hausdorff_distance
	elif name in ["manhattan", "manhattan_distance", "manhattan_func", "manhattan_func"]:
		return manhattan_distance
	else:
		printe(f"Wrong distance function name: {name}. Please check!")
		raise ValueError


def chebyshev_distance(array_x: np.ndarray, array_y: np.ndarray) -> float:
	""" Calculation of Chebyshev distance.
	"""
	n = array_x.shape[0]
	ret = -1 * np.inf
	for i in range(n):
		d = abs(array_x[i] - array_y[i])
		if d > ret:
			ret = d
	return ret


def cosine_distance(array_x: np.ndarray, array_y: np.ndarray) -> float:
	""" Calculation of Cosine distance.
	"""
	n = array_x.shape[0]
	xy_dot = 0.
	x_norm = 0.
	y_norm = 0.
	for i in range(n):
		xy_dot += array_x[i] * array_y[i]
		x_norm += array_x[i] * array_x[i]
		y_norm += array_y[i] * array_y[i]
	return 1. - xy_dot / (sqrt(x_norm) * sqrt(y_norm))


def euclidean_distance(array_x: np.ndarray, array_y: np.ndarray) -> float:
	""" Calculation of Euclidean distance.
	"""
	n   = array_x.shape[0]
	ret = 0.0
	for i in range(n):
		ret += (array_x[i] - array_y[i]) ** 2
	return sqrt(ret)


def hausdorff_distance(array_x: np.ndarray, array_y: np.ndarray) -> float:
	""" Calculation of Hausdorff distance.
	"""
	cmax = 0.0
	for i in range(len(array_x)):
		cmin = np.inf
		for j in range(len(array_y)):
			# euclidean_distance, manhattan_distance, chebyshev_distance, cosine_distance, haversine_distance
			d = euclidean_distance(array_x[i, :], array_y[j, :])
			if d < cmin:
				cmin = d
			if cmin < cmax:
				break
		if cmax < cmin < np.inf:
			cmax = cmin
	return cmax


def haversine_distance(array_x: np.ndarray, array_y: np.ndarray) -> float:
	""" Calculation of Haversine distance.
	"""
	R = 6378.0
	radians = np.pi / 180.0
	lat_x = radians * array_x[0]
	lon_x = radians * array_x[1]
	lat_y = radians * array_y[0]
	lon_y = radians * array_y[1]
	dlon  = lon_y - lon_x
	dlat  = lat_y - lat_x
	a = (pow(sin(dlat / 2.0), 2.0) + cos(lat_x) * cos(lat_y) * pow(sin(dlon / 2.0), 2.0))
	return R * 2 * asin(sqrt(a))


def manhattan_distance(array_x: np.ndarray, array_y: np.ndarray) -> float:
	""" Calculation of Manhattan distance.
	"""
	n = array_x.shape[0]
	ret = 0.
	for i in range(n):
		ret += abs(array_x[i] - array_y[i])
	return ret


# MARK: - Calculate Angle

def angle_between_arrays(array1: np.ndarray, array2: np.ndarray) -> float:
	""" Calculate angle of 2 trajectories
	"""
	vec1 = np.array([array1[-1][0] - array1[0][0], array1[-1][1] - array1[0][1]])
	vec2 = np.array([array2[-1][0] - array2[0][0], array2[-1][1] - array2[0][1]])
	
	L1 = np.sqrt(vec1.dot(vec1))
	L2 = np.sqrt(vec2.dot(vec2))
	
	if L1 == 0 or L2 == 0:
		return False
	
	cos = vec1.dot(vec2) / (L1 * L2)
	angle = np.arccos(cos) * 360 / (2 * np.pi)
	
	return angle
