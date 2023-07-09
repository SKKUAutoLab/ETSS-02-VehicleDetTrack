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
from .config import add_dirs_to_config
from .config import parse_config_from_json
from .config import parse_config_from_yaml
from .config import parse_config_to_namespace
from .config import process_config
from .device import select_device
from .dir import bin_dir
from .dir import create_dirs
from .dir import data_dir
from .dir import data_dir
from .dir import delete_files
from .dir import delete_files_matching
from .dir import list_files
from .dir import list_subdirs
from .dir import root_dir
from .dir import tests_dir
from .dir import tss_dir
from .dir import utils_dir
from .file import is_image_file
from .file import is_json_file
from .file import is_torch_saved_file
from .file import is_engine_saved_file
from .file import is_txt_file
from .file import is_video_file
from .file import is_video_stream
from .file import is_yaml_file
from .print import printe
from .print import prints
from .print import printw
