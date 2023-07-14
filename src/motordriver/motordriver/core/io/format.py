#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

# from ordered_enum import OrderedEnum
from enum import Enum

__all__ = [
	"ImageFormat", "VideoFormat"
]


# MARK: - ImageFormat

class ImageFormat(Enum):
	"""Define list of image format."""
	
	BMP  = ".bmp"
	DNG	 = ".dng"
	JPG  = ".jpg"
	JPEG = ".jpeg"
	PNG  = ".png"
	PPM  = ".ppm"
	TIF  = ".tif"
	TIFF = ".tiff"
	
	@staticmethod
	def values() -> list:
		"""Return the list of all image formats."""
		return [e.value for e in ImageFormat]
	

# MARK: - VideoFormat

class VideoFormat(Enum):
	"""Define list of video format."""
 
	AVI  = ".avi"
	M4V  = ".m4v"
	MKV  = ".mkv"
	MOV  = ".mov"
	MP4  = ".mp4"
	MPEG = ".mpeg"
	MPG  = ".mpg"
	WMV  = ".wmv"
	
	@staticmethod
	def values() -> list:
		"""Return the list of all video formats."""
		return [e.value for e in VideoFormat]
