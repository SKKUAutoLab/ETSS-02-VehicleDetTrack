#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
	def __init__(self):
		super(Exp, self).__init__()
		self.depth = 1.33
		self.width = 1.25
		self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
		self.input_size = (1280, 1280)  # (height, width)
		self.test_size  = (1280, 1280)

		# Define yourself dataset path
		self.data_dir  = "/media/vsw-ws-05/SSD_2/2_dataset/mtmc_dataset/46_SUWON/"
		self.train_ann = "annotations.json"
		self.val_ann   = "annotations.json"

		self.num_classes      = 3

		self.max_epoch        = 300
		self.data_num_workers = 8
		self.eval_interval    = 5

		# ---------- transform config ------------ #
		self.mosaic_prob = 1.0
		self.mixup_prob  = 1.0
		self.hsv_prob    = 1.0
		self.flip_prob   = 0.5
