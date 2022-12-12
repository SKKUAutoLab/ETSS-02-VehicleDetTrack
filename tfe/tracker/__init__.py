# ==================================================================== #
# File name: __init__.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/30/2021
#
# ``tracker`` API consists of several trackers that share the same interface.
# Hence, they can be swap easily.
# ==================================================================== #
from typing import Dict

from munch import Munch

from .tracker import Tracker


# MARK: - Lookup Table

def get_tracker(hparams: Dict, **kwargs) -> Tracker:
	"""Get the tracker based on the given hyperparameters.
	
	Args:
		hparams (dict):
			The tracker's hyperparameters.

	Returns:
		tracker (Tracker):
			The tracker.
	"""
	hparams = hparams if isinstance(hparams, Munch) else Munch(hparams)
	name    = hparams.name
	
	if name == "sort_kalman_bbox":
		from tfe.tracker.sort.sort_kalman_bbox import Sort
		return Sort(**hparams, **kwargs)
