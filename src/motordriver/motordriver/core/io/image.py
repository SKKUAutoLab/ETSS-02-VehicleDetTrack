#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import multiprocessing
import os
from glob import glob
from pathlib import Path
from typing import Optional
from typing import Union

import cv2
import numpy as np
# import pyvips
import torch
import torchvision
from joblib import delayed
from joblib import Parallel
from multipledispatch import dispatch
from PIL import ExifTags
from PIL import Image
from torch import Tensor

from core.io.filedir import create_dirs
from core.type.type import Arrays
from core.enhance.normalize import denormalize_naive
from core.utils.backend import interpolation_vision_backend_from_int
from core.utils.backend import interpolation_vision_backend_from_str
from core.utils.backend import VisionBackend
from core.utils.image import is_channel_first
from core.utils.image import to_channel_last
from core.io.format import ImageFormat

__all__ = [
	"exif_size",
	"read_image",
	"read_image_cv",
	"read_image_libvips",
	"read_image_pil",
	"ImageLoader",
	"ImageWriter",
	"is_bmp_file",
	"is_image_file",
	"write_image",
	"write_images"
]


# MARK: - Validate

def is_image_file(path: str) -> bool:
	"""Check if the given path is an image file."""
	if path is None:
		return False
	
	image_formats = ImageFormat.values()  # Acceptable image suffixes
	if os.path.isfile(path=path):
		extension = os.path.splitext(path.lower())[1]
		if extension in image_formats:
			return True
	return False


def is_bmp_file(path: str) -> bool:
	"""Check if the given path is a .bmp image file."""
	if path is None:
		return False
	
	bmp = ImageFormat.BMP.value  # Acceptable image suffixes
	if os.path.isfile(path=path):
		extension = os.path.splitext(path.lower())[1]
		if extension == bmp:
			return True
	return False


# MARK: - Read

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
	if ExifTags.TAGS[orientation] == "Orientation":
		break
		
		
# noinspection PyUnresolvedReferences,PyProtectedMember
def exif_size(image: Image) -> tuple:
	"""Return the exif-corrected PIL size."""
	size = image.size  # (width, height)
	try:
		rotation = dict(image._getexif().items())[orientation]
		if rotation == 6:  # rotation 270
			size = (size[1], size[0])
		elif rotation == 8:  # rotation 90
			size = (size[1], size[0])
	except:
		pass
	return size[1], size[0]


def read_image_cv(path: str) -> np.ndarray:
	"""Read image using opencv."""
	image = cv2.imread(path)   # BGR
	image = image[:, :, ::-1]  # BGR -> RGB
	image = np.ascontiguousarray(image)
	return image


def read_image_libvips(path: str) -> np.ndarray:
	"""Read image using libvips."""
	# image   = pyvips.Image.new_from_file(path, access="sequential")
	# mem_img = image.write_to_memory()
	# image   = np.frombuffer(mem_img, dtype=np.uint8).reshape(
	# 	image.height, image.width, 3
	# )
	# return image
	return None


def read_image_pil(path: str) -> np.ndarray:
	"""Read image using PIL."""
	image = Image.open(path)
	return np.asarray(image)


def read_image(
	path: str, backend: Union[VisionBackend, str, int] = "libvips"
) -> np.ndarray:
	"""Read image with the corresponding backend."""
	if isinstance(backend, int):
		backend = interpolation_vision_backend_from_int(backend)
	elif isinstance(backend, str):
		backend = interpolation_vision_backend_from_str(backend)
	
	if backend == VisionBackend.CV:
		return read_image_cv(path)
	elif backend == VisionBackend.LIBVIPS:
		return read_image_libvips(path)
	elif backend == VisionBackend.PIL:
		return read_image_pil(path)
	else:
		raise ValueError(f"Vision backend not supported: {backend}")
	

# MARK: - Write

# noinspection PyUnresolvedReferences
@dispatch(np.ndarray, str, str, str, str)
def write_image(
	image    : np.ndarray,
	dirpath  : str,
	name     : str,
	prefix   : str = "",
	extension: str = ".png"
):
	"""Save the image using `PIL`.

	Args:
		image (np.ndarray):
			A single image.
		dirpath (str):
			Fsaving directory.
		name (str):
			Name of the image file.
		prefix (str):
			Filename prefix. Default: ``.
		extension (str):
			Image file extension. One of [`.jpg`, `.jpeg`, `.png`, `.bmp`].
	"""
	if image.ndim not in [2, 3]:
		raise ValueError(f"Cannot save image with number of dimensions: "
						 f"{image.ndim}.")
	
	# NOTE: Unnormalize
	image = denormalize_naive(image)
	
	# NOTE: Convert to channel first
	if is_channel_first(image):
		image = to_channel_last(image)
	
	# NOTE: Convert to PIL image
	if not Image.isImageType(t=image):
		image = Image.fromarray(image.astype(np.uint8))
	
	# NOTE: Write image
	if create_dirs(paths=[dirpath]) == 0:
		base, ext = os.path.splitext(name)
		if ext:
			extension = ext
		if "." not in extension:
			extension = f".{extension}"
		if prefix in ["", None]:
			filepath = os.path.join(dirpath, f"{base}{extension}")
		else:
			filepath = os.path.join(dirpath, f"{prefix}_{base}{extension}")
		image.save(filepath)


@dispatch(Tensor, str, str, str, str)
def write_image(
	image    : Tensor,
	dirpath  : str,
	name     : str,
	prefix   : str = "",
	extension: str = ".png"
):
	"""Save the image using `torchvision`.

	Args:
		image (Tensor):
			A single image.
		dirpath (str):
			Fsaving directory.
		name (str):
			Name of the image file.
		prefix (str):
			Filename prefix. Default: ``.
		extension (str):
			Image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
	"""
	if image.dim() not in [2, 3]:
		raise ValueError(f"Cannot save image with number of dimensions: "
						 f"{image.dim()}.")
	
	# NOTE: Convert image
	image = denormalize_naive(image)
	image = to_channel_last(image)
	
	# NOTE: Write image
	if create_dirs(paths=[dirpath]) == 0:
		base, ext = os.path.splitext(name)
		if ext:
			extension = ext
		if "." not in extension:
			extension = f".{extension}"
		if prefix in ["", None]:
			filepath = os.path.join(dirpath, f"{base}{extension}")
		else:
			filepath = os.path.join(dirpath, f"{prefix}_{base}{extension}")
		
		if extension in [".jpg", ".jpeg"]:
			torchvision.io.image.write_jpeg(input=image, filename=filepath)
		elif extension in [".png"]:
			torchvision.io.image.write_png(input=image, filename=filepath)


@dispatch(np.ndarray, str, str, str)
def write_images(
	images   : np.ndarray,
	dirpath  : str,
	name     : str,
	extension: str = ".png"
):
	"""Save multiple images using `PIL`.

	Args:
		images (np.ndarray):
			A batch of images.
		dirpath (str):
			Fsaving directory.
		name (str):
			Name of the image file.
		extension (str):
			Image file extension. One of [`.jpg`, `.jpeg`, `.png`, `.bmp`].
	"""
	if images.ndim != 4:
		raise ValueError(f"Cannot save image with number of dimensions: "
						 f"{images.ndim}.")
	
	num_jobs = multiprocessing.cpu_count()
	Parallel(n_jobs=num_jobs)(
		delayed(write_image)(image, dirpath, name, f"{index}", extension)
		for index, image in enumerate(images)
	)


@dispatch(Tensor, str, str, str)
def write_images(
	images   : Tensor,
	dirpath  : str,
	name     : str,
	extension: str = ".png"
):
	"""Save multiple images using `torchvision`.

	Args:
		images (Tensor):
			A image of image.
		dirpath (str):
			Fsaving directory.
		name (str):
			Name of the image file.
		extension (str):
			Image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
	"""
	if images.dim() != 4:
		raise ValueError(f"Cannot save image with number of dimensions: "
						 f"{images.dim()}.")
	
	num_jobs = multiprocessing.cpu_count()
	Parallel(n_jobs=num_jobs)(
		delayed(write_image)(image, dirpath, name, f"{index}", extension)
		for index, image in enumerate(images)
	)


@dispatch(list, str, str, str)
def write_images(
	images   : list,
	dirpath  : str,
	name     : str,
	extension: str = ".png"
):
	"""Save multiple images.

	Args:
		images (list):
			A list of images.
		dirpath (str):
			Fsaving directory.
		name (str):
			Name of the image file.
		extension (str):
			Image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
	"""
	if (isinstance(images, list) and
		all(isinstance(image, np.ndarray) for image in images)):
		cat_image = np.concatenate([images], axis=0)
		write_images(cat_image, dirpath, name, extension)
	elif (isinstance(images, list) and
		  all(torch.is_tensor(image) for image in images)):
		cat_image = torch.stack(images)
		write_images(cat_image, dirpath, name, extension)
	else:
		raise TypeError(f"Cannot concatenate images of type: {type(images)}.")


@dispatch(dict, str, str, str)
def write_images(
	images   : dict,
	dirpath  : str,
	name     : str,
	extension: str = ".png"
):
	"""Save multiple images.

	Args:
		images (dict):
			A list of images.
		dirpath (str):
			Fsaving directory.
		name (str):
			Name of the image file.
		extension (str):
			Image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
	"""
	if (isinstance(images, dict) and
		all(isinstance(image, np.ndarray) for _, image in images.items())):
		cat_image = np.concatenate([image for key, image in images.items()],
								   axis=0)
		write_images(cat_image, dirpath, name, extension)
	elif (isinstance(images, dict) and
		  all(torch.is_tensor(image) for _, image in images)):
		values    = list(tuple(images.values()))
		cat_image = torch.stack(values)
		write_images(cat_image, dirpath, name, extension)
	else:
		raise TypeError


# MARK: - ImageLoader/Writer

class ImageLoader:
	"""Image Loader retrieves and loads image(s) from a filepath, a pathname
	pattern, or directory.

	Attributes:
		data (str):
			Data source. Can be a path to an image file or a directory.
			It can be a pathname pattern to images.
		batch_size (int):
			Number of samples in one forward & backward pass.
		image_files (list):
			List of image files found in the data source.
		num_images (int):
			Total number of images.
		index (int):
			Current index.
	"""

	# MARK: Magic Functions

	def __init__(self, data: str, batch_size: int = 1):
		super().__init__()
		self.data        = data
		self.batch_size  = batch_size
		self.image_files = []
		self.num_images  = -1
		self.index       = 0
		
		self.init_image_files(data=self.data)

	def __len__(self):
		"""Return the number of images in the `image_files`."""
		return self.num_images  # Number of images
	
	def __iter__(self):
		"""Return an iterator starting at index 0."""
		self.index = 0
		return self

	def __next__(self):
		"""Next iterator.
		
		Examples:
			>>> video_stream = ImageLoader("cam_1.mp4")
			>>> for index, image in enumerate(video_stream):
		
		Returns:
			images (np.ndarray):
				List of image file from opencv with `np.ndarray` type.
			indexes (list):
				List of image indexes.
			files (list):
				List of image files.
			rel_paths (list):
				List of images' relative paths corresponding to data.
		"""
		if self.index >= self.num_images:
			raise StopIteration
		else:
			images    = []
			indexes   = []
			files     = []
			rel_paths = []

			for i in range(self.batch_size):
				if self.index >= self.num_images:
					break
				
				file     = self.image_files[self.index]
				rel_path = file.replace(self.data, "")

				images.append(cv2.imread(self.image_files[self.index]))
				indexes.append(self.index)
				files.append(file)
				rel_paths.append(rel_path)

				self.index += 1

			return np.array(images), indexes, files, rel_paths
	
	# MARK: Configure
	
	def init_image_files(self, data: str):
		"""Initialize list of image files in data source.
		
		Args:
			data (str):
				Data source. Can be a path to an image file or a directory.
				It can be a pathname pattern to images.
		"""
		if is_image_file(data):
			self.image_files = [data]
		elif os.path.isdir(data):
			self.image_files = [
				img for img in glob(os.path.join(data, "**/*"), recursive=True)
				if is_image_file(img)
			]
		elif isinstance(data, str):
			self.image_files = [img for img in glob(data) if is_image_file(img)]
		else:
			raise IOError("Error when reading input image files.")
		self.num_images = len(self.image_files)

	def list_image_files(self, data: str):
		"""Alias of `init_image_files()`."""
		self.init_image_files(data=data)


class ImageWriter:
	"""Video Writer saves images to a destination directory.

	Attributes:
		dst (str):
			Output directory or filepath.
		extension (str):
			Image file extension. One of [`.jpg`, `.jpeg`, `.png`, `.bmp`].
		index (int):
			Current index. Default: `0`.
	"""

	# MARK: Magic Functions

	def __init__(self, dst: str, extension: str = ".jpg"):
		super().__init__()
		self.dst	   = dst
		self.extension = extension
		self.index     = 0

	def __len__(self):
		"""Return the number of already written images."""
		return self.index

	# MARK: Write

	def write_image(self, image: np.ndarray, image_file: Optional[str] = None):
		"""Write image.

		Args:
			image (np.ndarray):
				Image.
			image_file (str, optional):
				Image file.
		"""
		if is_channel_first(image):
			image = to_channel_last(image)
		
		if image_file is not None:
			image_file = (image_file[1:] if image_file.startswith("\\")
						  else image_file)
			image_file = (image_file[1:] if image_file.startswith("/")
						  else image_file)
			image_name = os.path.splitext(image_file)[0]
		else:
			image_name = f"{self.index}"
		
		output_file = os.path.join(self.dst, f"{image_name}{self.extension}")
		parent_dir  = str(Path(output_file).parent)
		create_dirs(paths=[parent_dir])
		
		cv2.imwrite(output_file, image)
		self.index += 1

	def write_images(
		self, images: Arrays, image_files: Optional[list[str]] = None
	):
		"""Write batch of images.

		Args:
			images (Arrays):
				Images.
			image_files (list[str], optional):
				Image files.
		"""
		if image_files is None:
			image_files = [None for _ in range(len(images))]
		# assert len(image_files) == len(images), \
		# 	f"{len(image_files)} != {len(images)}"

		for image, image_file in zip(images, image_files):
			self.write_image(image=image, image_file=image_file)


