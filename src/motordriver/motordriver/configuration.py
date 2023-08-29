#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
from typing import Union

from munch import Munch

from core.io.file import load


# MARK: - Directories

if "DIR_TSS" in os.environ:
    root_dir   = os.environ["DIR_TSS"]
else:
    root_dir   = os.path.dirname(os.path.abspath(__file__))  	    # "src/motor"

src_dir        = os.path.join(root_dir, "motor")                	# "src/motor/motor"
config_dir     = os.path.join(root_dir, "configs")              	# "src/motor/configs"
models_zoo_dir = os.path.join(root_dir, "models")        			# "src/motor/models_zoo"
if "DIR_DATA" in os.environ:
    data_dir     = os.environ["DIR_DATA"]
    result_dir   = os.environ["DIR_RESULT"]
else:
    data_dir     = os.path.join(root_dir, "data")              		# "src/motor/data"
    result_dir   = os.path.join(root_dir, "data")              		# "src/motor/data"

# MARK: - Process Config


def load_config(config: Union[str, dict]) -> Munch:
	"""Load and process config from file.

	Args:
		config (str, dict):
			Config filepath that contains configuration values or the
			config dict.

	Returns:
		config (Munch):
			Config dictionary as namespace.
	"""
	# NOTE: Load dictionary from file and convert to namespace using Munch
	if isinstance(config, str):
		config_dict = load(path=config)
	elif isinstance(config, dict):
		config_dict = config
	else:
		raise ValueError

	assert config_dict is not None, f"No configuration is found at {config}!"
	config = Munch.fromDict(config_dict)
	return config
