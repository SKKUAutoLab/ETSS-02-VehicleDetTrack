# ==================================================================== #
# File name: __init__.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 29/08/2023
#
# ``analyses`` API consists of several analyses that share the same interface.
# Hence, they can be swap easily.
# ==================================================================== #

from __future__ import annotations

from .analyzer import BaseAnalyzer
from .driver_violation import DriverViolation
