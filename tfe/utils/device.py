# ==================================================================== #
# File name: device.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/27/2021
# ==================================================================== #
import os

import torch

from .print import prints


def select_device(
	model_name: str = "",
	device    : str = "",
	batch_size: int = None
):
	"""Select the device to run the model.
	"""
	# device = 'cpu' or '0' or '0,1,2,3'
	s = f"{model_name}"  # string
	cpu = device.lower() == "cpu"
	
	if cpu:
		os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
	elif device:  # non-cpu device requested
		os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
		assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"  # check availability
	
	cuda = not cpu and torch.cuda.is_available()
	
	if cuda:
		n = torch.cuda.device_count()
		
		if n > 1 and batch_size:  # check that batch_size is compatible with device_count
			assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
		space = " " * len(s)
		
		for i, d in enumerate(device.split(",") if device else range(n)):
			p = torch.cuda.get_device_properties(i)
			s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
	else:
		s += 'CPU\n'
	
	prints(s)  # skip a line
	return torch.device("cuda:0" if cuda else "cpu")
