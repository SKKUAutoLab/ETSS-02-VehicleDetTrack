# ==================================================================== #
# Copyright (C) 2022 - Automation Lab - Sungkyunkwan University
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ``Tracker`` base class for all variant of tracker.
# It define a unify template to guarantee the input and output of all tracker are the same.
# Usually, each ``Tracker`` class is associate with a ``Track`` class
#
# Subclassing guide:
# 1. The package (i.e, the .py filename) should be in the template:
#    {tracker}_{track_motion_model}_{feature_used_to_track}
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
