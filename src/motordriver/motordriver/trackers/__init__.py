# ==================================================================== #
# File name: __init__.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 19/07/2023
#
# ``tracker`` API consists of several trackers that share the same interface.
# Hence, they can be swap easily.
# ==================================================================== #
from typing import Dict

from munch import Munch

from .tracker import Tracker
from .sort.sort_kalman_bbox import Sort
