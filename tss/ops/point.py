# ==================================================================== #
# File name: track.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/29/2021
#
# Implements common operations on point
# ==================================================================== #
from math import sqrt

import numpy as np


# MARK: - Calculate Distance

def distance_between_points(
	point_a: np.ndarray,
	point_b: np.ndarray
) -> float:
	""" Calculate Euclidean distance between TWO points.
	
	Args:
		point_a (np.ndarray):
			The first points.
		point_b(np.ndarray):
			The second points.

	Returns:
		(float)
			The distance.
	"""
	return sqrt(((point_a[0] - point_b[0]) ** 2) + ((point_a[1] - point_b[1]) ** 2))
