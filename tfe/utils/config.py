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
import json
import os
from typing import Dict
from typing import Optional

import yaml
from munch import Munch

from . import dir


# MARK: Process Config

def process_config(config_path: str) -> Munch:
    """Process the config file that contains the model configurations.
    """
    # NOTE: Parse the configurations from the config json file provided
    config = parse_config_to_namespace(config_path=config_path)

    # NOTE: Add sub-directory paths
    config = add_dirs_to_config(config=config)
    return config


def parse_config_to_namespace(config_path: str) -> Optional[Dict]:
    """Parse config from file (json or yaml) to namespace.
    """
    _, file_extension = os.path.splitext(config_path)
    config_dict = None
    if file_extension == ".json":
        config_dict = parse_config_from_json(json_path=config_path)
    elif file_extension == ".yaml":
        config_dict = parse_config_from_yaml(yaml_path=config_path)

    if config_dict is None:
        return None

    # NOTE: Convert the dictionary to a namespace using Munch
    config = Munch.fromDict(config_dict)
    return config


def parse_config_from_json(json_path: str) -> Dict:
    """Get the config values from the json file (should be found in the "configs" folder).
    """
    with open(json_path, "r") as config_file:
        config_dict = json.load(config_file)
    return config_dict


def parse_config_from_yaml(yaml_path: str) -> Dict:
    """Get the config values from the yaml file (should be found in the "configs" folder).
    """
    with open(yaml_path, "r") as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
    return config_dict


def add_dirs_to_config(config: Dict) -> Munch:
    """Define necessary dir paths to the config dictionary.
    """
    # NOTE: Define output_dir
    data_output_dir   = os.path.join(dir.data_dir, config.data.dataset, "outputs")
    camera_output_dir = os.path.join(data_output_dir, config.camera_name)
    
    # NOTE: Add dirs to config
    config.dirs = Munch()
    config.dirs.data_output_dir   = data_output_dir
    config.dirs.camera_output_dir = camera_output_dir
    
    # NOTE: Create dirs
    dir.create_dirs(
        dirs=[
            config.dirs.data_output_dir,
            config.dirs.camera_output_dir
            ]
    )
    
    return config
