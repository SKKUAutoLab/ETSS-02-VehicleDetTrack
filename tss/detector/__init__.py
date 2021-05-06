# ==================================================================== #
# File name: __init__.py
# Author: Long H. Pham and Duong N.-N. Tran
# Date created: 03/27/2021
#
# ``detector`` API consists of several detector models that share the same interface.
# Hence, they can be swap easily.
# ==================================================================== #
from typing import Dict

from munch import Munch

from .detection import Detection
from .detector import Detector
from .yolov5 import YOLOv5
from .yolov5 import YOLOv5TRT

# MARK: - Lookup Table

def get_detector(hparams: Dict, **kwargs) -> Detector:
	"""Get the detector model based on the given hyperparameters.
	
	Args:
		hparams (dict):
			The model's hyperparameters.

	Returns:
		detector (Detector):
			The detector model.
	"""
	hparams = hparams if isinstance(hparams, Munch) else Munch(hparams)
	name    = hparams.name
	
	if name == "yolov5":  # detector :: name
		return YOLOv5(**hparams, **kwargs)
	elif name == "yolov5trt":
		return YOLOv5TRT(**hparams, **kwargs)
