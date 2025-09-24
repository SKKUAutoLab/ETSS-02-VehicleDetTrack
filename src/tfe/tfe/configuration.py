#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
from typing import Union

from munch import Munch

from tfe.io.file import load

# MARK: - Directories

if "ROOT_DIR" in os.environ:
    root_dir   = os.environ["ROOT_DIR"]
else:
    root_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  	    # "src/tfe"

data_dir       = os.path.join(root_dir, "data")                  	# "src/tfe/data"
models_zoo_dir = os.path.join(root_dir, "model_zoo")        			# "src/tfe/models_zoo"


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
