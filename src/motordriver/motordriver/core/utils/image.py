# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional
from typing import Union

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torchvision
from multipledispatch import dispatch
from torch import Tensor
from torchvision.transforms.functional import _is_numpy, _is_pil_image

from core.type.collection import is_list_of
from core.type.type import Size2T
from core.utils.rich import error_console

from core.factory.builder import TRANSFORMS

__all__ = [
	"get_image_size",
	"image_channels",
	"image_list_to_tensor",
	"ImageListToTensor",
	"integer_to_color",
	"is_channel_first",
	"is_channel_last",
	"is_color_image",
	"is_integer_image",
	"is_one_hot_image",
	"labels_to_one_hot",
	"tensor_list_to_image",
	"TensorListToImage",
	"to_4d_image_list",
	"to_channel_first",
	"to_channel_last",
	"to_image",
	"to_pil_image",
	"to_tensor",
	"ToImage",
	"ToTensor",
]


# MARK: - Image Encode

def is_color_image(image: Union[Tensor, np.ndarray]) -> bool:
	"""Check if the given image is color encoded."""
	c = image_channels(image)
	if c == 3 or c == 4:
		return True
	return False


def is_integer_image(image: Union[Tensor, np.ndarray]) -> bool:
	"""Check if the given image is integer-encoded."""
	c = image_channels(image)
	if c == 1:
		return True
	return False


def is_one_hot_image(image: Union[Tensor, np.ndarray]) -> bool:
	"""Check if the given image is one-hot encoded."""
	c = image_channels(image)
	if c > 1:
		return True
	return False


def _integer_to_color(image: np.ndarray, colors: list) -> np.ndarray:
	"""Convert the integer-encoded image to color image. Fill an image with
	labels' colors.

	Args:
		image (np.ndarray):
			An image in either one-hot or integer.
		colors (list):
			List of all colors.

	Returns:
		color (np.ndarray):
			Colored image.
	"""
	if len(colors) <= 0:
		raise ValueError(f"No colors are provided.")

	# NOTE: Convert to channel-first
	image = to_channel_first(image)

	# NOTE: Squeeze dims to 2
	if image.ndim == 3:
		image = np.squeeze(image)

	# NOTE: Draw color
	r = np.zeros_like(image).astype(np.uint8)
	g = np.zeros_like(image).astype(np.uint8)
	b = np.zeros_like(image).astype(np.uint8)
	for l in range(0, len(colors)):
		idx = image == l
		r[idx] = colors[l][0]
		g[idx] = colors[l][1]
		b[idx] = colors[l][2]
	rgb = np.stack([r, g, b], axis=0)
	return rgb


@dispatch(Tensor, list)
def integer_to_color(image: Tensor, colors: list) -> Tensor:
	mask_np = image.numpy()
	mask_np = integer_to_color(mask_np, colors)
	color   = torch.from_numpy(mask_np)
	return color


@dispatch(np.ndarray, list)
def integer_to_color(image: np.ndarray, colors: list) -> np.ndarray:
	# NOTE: If [C, H, W]
	if image.ndim == 3:
		return _integer_to_color(image, colors)

	# NOTE: If [B, C, H, W]
	if image.ndim == 4:
		colors = [_integer_to_color(i, colors) for i in image]
		colors = np.stack(colors).astype(np.uint8)
		return colors

	raise ValueError(f"Wrong image's dimension: {image.ndim}.")


def labels_to_one_hot(
		labels     : Tensor,
		num_classes: int,
		device     : Optional[torch.device] = None,
		dtype      : Optional[torch.dtype]  = None,
		eps        : float                  = 1e-6,
) -> Tensor:
	"""Convert an integer label x-D image to a one-hot (x+1)-D image.

	Args:
		labels (Tensor):
			Tensor with labels of shape [N, *], where N is batch size.
			Each value is an integer representing correct classification.
		num_classes (int):
			Number of classes in labels.
		device (torch.device, optional):
			Desired device of returned image.
		dtype (torch.dtype):
			Desired data type of returned image.
		eps (float)

	Returns:
		one_hot (Tensor):
			Labels in one hot image of shape [N, C, *].

	Examples:
		>>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
		>>> one_hot(labels, num_classes=3)
		image([[[[1.0000e+00, 1.0000e-06],
				  [1.0000e-06, 1.0000e+00]],
		<BLANKLINE>
				 [[1.0000e-06, 1.0000e+00],
				  [1.0000e-06, 1.0000e-06]],
		<BLANKLINE>
				 [[1.0000e-06, 1.0000e-06],
				  [1.0000e+00, 1.0000e-06]]]])
	"""
	if not isinstance(labels, Tensor):
		raise TypeError(f"Input labels type is not a Tensor. "
						f"Got: {type(labels)}")
	if not labels.dtype == torch.int64:
		raise ValueError(f"labels must be of the same dtype torch.int64. "
						 f"Got: {labels.dtype}")
	if num_classes < 1:
		raise ValueError("Number of classes must be bigger than one. "
						 "Got: {}".format(num_classes))

	shape   = labels.shape
	one_hot = torch.zeros((shape[0], num_classes) + shape[1:],
						  device=device, dtype=dtype)
	return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


# MARK: - Image Format

@dispatch(Tensor)
def is_channel_first(image: Tensor) -> bool:
	"""Check if the image is in channel first format."""
	if image.ndim == 5:
		_, _, s2, s3, s4 = [int(x) for x in image.shape]
		if (s2 < s3) and (s2 < s4):
			return True
		elif (s4 < s2) and (s4 < s3):
			return False

	if image.ndim == 4:
		_, s1, s2, s3 = [int(x) for x in image.shape]
		if (s1 < s2) and (s1 < s3):
			return True
		elif (s3 < s1) and (s3 < s2):
			return False

	if image.ndim == 3:
		s0, s1, s2 = [int(x) for x in image.shape]
		if (s0 < s1) and (s0 < s2):
			return True
		elif (s2 < s0) and (s2 < s1):
			return False

	raise ValueError(f"Cannot process image with shape {image.shape}.")


@dispatch(np.ndarray)
def is_channel_first(image: np.ndarray) -> bool:
	"""Check if the image is in channel first format."""
	if image.ndim == 5:
		_, _, s2, s3, s4 = image.shape
		if (s2 < s3) and (s2 < s4):
			return True
		elif (s4 < s2) and (s4 < s3):
			return False

	if image.ndim == 4:
		_, s1, s2, s3 = image.shape
		if (s1 < s2) and (s1 < s3):
			return True
		elif (s3 < s1) and (s3 < s2):
			return False

	if image.ndim == 3:
		s0, s1, s2 = image.shape
		if (s0 < s1) and (s0 < s2):
			return True
		elif (s2 < s0) and (s2 < s1):
			return False

	raise ValueError(f"Cannot process image with shape {image.shape}.")


def is_channel_last(image: Union[Tensor, np.ndarray]) -> bool:
	"""Check if the image is in channel last format."""
	return not is_channel_first(image)


@dispatch(Tensor)
def image_channels(image: Tensor) -> int:
	"""Get number of channels of the image."""
	if image.ndim == 4:
		if is_channel_first(image):
			_, c, h, w = [int(x) for x in image.shape]
		else:
			_, h, w, c = [int(x) for x in image.shape]
		return c

	if image.ndim == 3:
		if is_channel_first(image):
			c, h, w = [int(x) for x in image.shape]
		else:
			h, w, c = [int(x) for x in image.shape]
		return c

	raise ValueError(f"Cannot process image with shape {image.shape}.")


@dispatch(np.ndarray)
def image_channels(image: np.ndarray) -> int:
	"""Get number of channels of the image."""
	if image.ndim == 4:
		if is_channel_first(image):
			_, c, h, w = image.shape
		else:
			_, h, w, c = image.shape
		return c

	if image.ndim == 3:
		if is_channel_first(image):
			c, h, w = image.shape
		else:
			h, w, c = image.shape
		return c

	raise ValueError(f"Cannot process image with shape {image.shape}.")


@dispatch(Tensor, keep_dim=bool)
def to_channel_first(image: Tensor, keep_dim: bool = True) -> Tensor:
	"""Convert image to channel first format.

	Args:
		keep_dim (bool):
			If `False` unsqueeze the image to match the shape [B, H, W, C].
			Default: `True`.
	"""
	image = deepcopy(image)
	if is_channel_first(image):
		pass
	elif image.ndim == 2:
		image    = image.unsqueeze(0)
	elif image.ndim == 3:
		image    = image.permute(2, 0, 1)
	elif image.ndim == 4:
		image    = image.permute(0, 3, 1, 2)
		keep_dim = True
	elif image.ndim == 5:
		image    = image.permute(0, 1, 4, 2, 3)
		keep_dim = True
	else:
		raise ValueError(f"Cannot process image with shape {image.shape}.")

	return image.unsqueeze(0) if not keep_dim else image


@dispatch(np.ndarray, keep_dim=bool)
def to_channel_first(image: np.ndarray, keep_dim: bool = True) -> np.ndarray:
	"""Convert image to channel first format.

	Args:
		keep_dim (bool):
			If `False` unsqueeze the image to match the shape [B, H, W, C].
			Default: `True`.
	"""
	image = deepcopy(image)
	if is_channel_first(image):
		pass
	elif image.ndim == 2:
		image    = np.expand_dims(image, 0)
	elif image.ndim == 3:
		image    = np.transpose(image, (2, 0, 1))
	elif image.ndim == 4:
		image    = np.transpose(image, (0, 3, 1, 2))
		keep_dim = True
	elif image.ndim == 5:
		image    = np.transpose(image, (0, 1, 4, 2, 3))
		keep_dim = True
	else:
		raise ValueError(f"Cannot process image with shape {image.shape}.")

	return np.expand_dims(image, 0) if not keep_dim else image


@dispatch(Tensor, keep_dim=bool)
def to_channel_last(image: Tensor, keep_dim: bool = True) -> Tensor:
	"""Convert image to channel last format.

	Args:
		keep_dim (bool):
			If `False` squeeze the input image to match the shape [H, W, C] or
			[H, W]. Default: `True`.
	"""
	image       = deepcopy(image)
	input_shape = image.shape

	if is_channel_last(image):
		pass
	elif image.ndim == 2:
		pass
	elif image.ndim == 3:
		if input_shape[0] == 1:
			# Grayscale for proper plt.imshow needs to be [H, W]
			image = image.squeeze()
		else:
			image = image.permute(1, 2, 0)
	elif image.ndim == 4:  # [B, C, H, W] -> [B, H, W, C]
		image = image.permute(0, 2, 3, 1)
		if input_shape[0] == 1 and not keep_dim:
			image = image.squeeze(0)
		if input_shape[1] == 1:
			image = image.squeeze(-1)
	elif image.ndim == 5:
		image = image.permute(0, 1, 3, 4, 2)
		if input_shape[0] == 1 and not keep_dim:
			image = image.squeeze(0)
		if input_shape[2] == 1:
			image = image.squeeze(-1)
	else:
		raise ValueError(f"Cannot process image with shape {image.shape}.")

	return image


@dispatch(np.ndarray, keep_dim=bool)
def to_channel_last(image: np.ndarray, keep_dim: bool = True) -> np.ndarray:
	"""Convert image to channel last format.

	Args:
		keep_dim (bool):
			If `False` squeeze the input image to match the shape [H, W, C] or
			[H, W]. Default: `True`.
	"""
	image       = deepcopy(image)
	input_shape = image.shape

	if is_channel_last(image):
		pass
	elif image.ndim == 2:
		pass
	elif image.ndim == 3:
		if input_shape[0] == 1:
			# Grayscale for proper plt.imshow needs to be [H, W]
			image = image.squeeze()
		else:
			image = np.transpose(image, (1, 2, 0))
	elif image.ndim == 4:
		image = np.transpose(image, (0, 2, 3, 1))
		if input_shape[0] == 1 and not keep_dim:
			image = image.squeeze(0)
		if input_shape[1] == 1:
			image = image.squeeze(-1)
	elif image.ndim == 5:
		image = np.transpose(image, (0, 1, 3, 4, 2))
		if input_shape[0] == 1 and not keep_dim:
			image = image.squeeze(0)
		if input_shape[2] == 1:
			image = image.squeeze(-1)
	else:
		raise ValueError(f"Cannot process image with shape {image.shape}.")

	return image


# MARK: - Image Info

def get_image_size(image: Union[Tensor, np.ndarray, PIL.Image]) -> Size2T:
	"""Returns the size of an image as [width, height].

	Args:
		image (Tensor, np.ndarray, PIL Image):
			The image to be checked.

	Returns:
		(Size2T):
			The image size as [W, H].
	"""
	if isinstance(image, (Tensor, np.ndarray)):
		if is_channel_first(image):  # [.., C, H, W]
			return [image.shape[-1], image.shape[-2]]
		else:  # [.., H, W, C]
			return [image.shape[-2], image.shape[-3]]
	elif _is_pil_image(image):
		return list(image.size)
	else:
		TypeError(f"Unexpected type {type(image)}")


# MARK: - To Tensor

def to_tensor(
		image    : Union[np.ndarray, PIL.Image],
		keep_dim : bool = True,
		normalize: bool = False,
) -> Tensor:
	"""Convert a `PIL Image` or `np.ndarray` image to a 4d tensor.

	Args:
		image (np.ndarray, PIL.Image):
			Image of the form [H, W, C], [H, W] or [B, H, W, C].
		keep_dim (bool):
			If `False` unsqueeze the image to match the shape [B, H, W, C].
			Default: `True`.
		normalize (bool):
			If `True`, converts the tensor in the range [0, 255] to the range
			[0.0, 1.0]. Default: `False`.

	Returns:
		(Tensor):
			Converted image.
	"""
	if not (_is_pil_image(image) or _is_numpy(image) or torch.is_tensor(image)):
		raise TypeError(f"image should be PIL Image, ndarray, or Tensor. "
						f"Got: {type(image)}.")

	if ((_is_numpy(image) or torch.is_tensor(image))
			and (image.ndim > 4 or image.ndim < 2)):
		raise ValueError(f"image should be 2/3/4 dimensional. "
						 f"Got: {image.ndim} dimensions.")

	# _image = image
	_image = deepcopy(image)

	# NOTE: Handle PIL Image
	if _is_pil_image(_image):
		mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
		_image = np.array(
			_image, mode_to_nptype.get(_image.mode, np.uint8), copy=True
		)
		if image.mode == "1":
			_image = 255 * _image

	# NOTE: Handle numpy array
	if _is_numpy(_image):
		_image = torch.from_numpy(_image).contiguous()

	# NOTE: Channel first format
	_image = to_channel_first(_image, keep_dim=keep_dim)

	# NOTE: Normalize
	if normalize:
		from core.enhance import normalize_naive
		_image = normalize_naive(_image)

	if isinstance(_image, torch.ByteTensor):
		return _image.to(dtype=torch.get_default_dtype())
	return _image


def image_list_to_tensor(
		images   : list[Union[np.ndarray, PIL.Image]],
		keep_dim : bool = True,
		normalize: bool = False,
) -> Tensor:
	"""Converts a list of `PIL Image` or `np.ndarray` images to a 4d tensor.

	Args:
		images (list):
			List of images, each of the form [H, W] or [H, W, C].
			Image shapes must be consistent.
		keep_dim (bool):
			If `False` unsqueeze the image to match the shape [B, H, W, C].
			Default: `True`.
		normalize (bool):
			If `True`, converts the tensor in the range [0, 255] to the range
			[0.0, 1.0]. Default: `False`.

	Returns:
		(Tensor):
			Tensor of the form [B, H, W, C].

	Example:
		>>> images = [np.ones((4, 4, 1)), np.zeros((4, 4, 1))]
		>>> image_list_to_tensor(images).shape
		torch.Size([2, 1, 4, 4])
	"""
	if not images:
		raise ValueError("Input list is empty.")

	tensors = [to_tensor(i, keep_dim, normalize) for i in images]
	return torch.stack(tensors)


@TRANSFORMS.register(name="to_tensor", force=True)
class ToTensor(nn.Module):
	"""Convert a `PIL Image` or `np.ndarray` image to a 4d tensor.

	Args:
		keep_dim (bool):
			If `False` unsqueeze the image to match the shape [B, H, W, C].
			Default: `True`.
		normalize (bool):
			If `True`, converts the tensor in the range [0, 255] to the range
			[0.0, 1.0]. Default: `False`.
	"""

	def __init__(self, keep_dim: bool = False, normalize: bool = False):
		super().__init__()
		self.keep_dim  = keep_dim
		self.normalize = normalize

	def forward(self, image: Union[np.ndarray, PIL.Image]) -> Tensor:
		return to_tensor(image, self.keep_dim, self.normalize)


@TRANSFORMS.register(name="image_list_to_tensor")
class ImageListToTensor(nn.Module):
	"""Converts a list of `PIL Image` or `np.ndarray` images to a 4d tensor.

	Args:
		keep_dim (bool):
			If `False` unsqueeze the image to match the shape [B, H, W, C].
			Default: `True`.
		normalize (bool):
			If `True`, converts the tensor in the range [0, 255] to the range
			[0.0, 1.0]. Default: `False`.
	"""

	def __init__(self, keep_dim: bool = False, normalize: bool = False):
		super().__init__()
		self.keep_dim  = keep_dim
		self.normalize = normalize

	def forward(self, image: list[Union[np.ndarray, PIL.Image]]) -> Tensor:
		return image_list_to_tensor(image, self.keep_dim, self.normalize)


# MARK: - To Image

def to_image(
		tensor: Tensor, keep_dim: bool = True, denormalize: bool = False
) -> np.ndarray:
	"""Converts a PyTorch tensor to a numpy image. In case the image is in the
	GPU, it will be copied back to CPU.

	Args:
		tensor (Tensor):
			Image of the form [H, W], [C, H, W] or [B, H, W, C].
		keep_dim (bool):
			If `False` squeeze the input image to match the shape [H, W, C] or
			[H, W]. Default: `True`.
		denormalize (bool):
			If `True`, converts the image in the range [0.0, 1.0] to the range
			[0, 255]. Default: `False`.

	Returns:
		image (np.ndarray):
			Image of the form [H, W], [H, W, C] or [B, H, W, C].

	Example:
		>>> img = torch.ones(1, 3, 3)
		>>> to_image(img).shape
		(3, 3)

		>>> img = torch.ones(3, 4, 4)
		>>> to_image(img).shape
		(4, 4, 3)
	"""
	if not torch.is_tensor(tensor):
		error_console.log(f"Input type is not a Tensor. Got: {type(tensor)}.")
		return tensor
	if len(tensor.shape) > 4 or len(tensor.shape) < 2:
		raise ValueError(f"Input size must be a 2/3/4 dimensional tensor. "
						 f"Got: {tensor.shape}")

	image = tensor.cpu().detach().numpy()

	# NOTE: Channel last format
	image = to_channel_last(image, keep_dim=keep_dim)

	# NOTE: Denormalize
	if denormalize:
		from core.enhance import denormalize_naive
		image = denormalize_naive(image)

	return image.astype(np.uint8)


def tensor_list_to_image(
		tensors: list[Tensor], keep_dim: bool = True, denormalize: bool = False
) -> np.ndarray:
	"""Converts a list of tensors to a numpy image.

	Args:
		tensors (ListOrTupleAnyT[Tensor]):
			List of tensors, each of the form [H, W, C]. Image shapes must be
			consistent.
		keep_dim (bool):
			If `False` squeeze the input image to match the shape [H, W, C] or
			[H, W]. Default: `True`.
		denormalize (bool):
			If `True`, converts the image in the range [0.0, 1.0] to the range
			[0, 255]. Default: `False`.

	Returns:
		image (np.ndarray):
			Tensor of the form [B, H, W, C].

	Example:
		>>> images = [np.ones((4, 4, 1)), np.zeros((4, 4, 1))]
		>>> image_list_to_tensor(images).shape
		torch.Size([2, 1, 4, 4])
	"""
	if not tensors:
		raise ValueError("Input list is empty.")

	images = [to_image(t, keep_dim, denormalize) for t in tensors]
	return np.stack(images)


def to_4d_image_list(x: Union[Tensor, np.ndarray]) -> list[np.ndarray]:
	"""Convert to a 4D-array list."""
	if isinstance(x, Tensor):
		x = x.detach().cpu().numpy()

	if isinstance(x, np.ndarray):
		if x.ndim < 3:
			raise ValueError(f"Wrong dimension: x.ndim < 3.")
		elif x.ndim == 3:
			x = [np.expand_dims(x, axis=0)]
		elif x.ndim == 4:
			x = [x]
		elif x.ndim == 5:
			x = list(x)
		return x

	if isinstance(x, tuple):
		x = list(x)

	if isinstance(x, dict):
		x = [v for k, v in x.items()]

	if isinstance(x, list) and is_list_of(x, Tensor):
		x = [_x.detach().cpu().numpy() for _x in x]

	if isinstance(x, list) and is_list_of(x, np.ndarray):
		if all(x_.ndim < 3 for x_ in x):
			raise ValueError(f"Wrong dimension: x.ndim < 3.")
		elif all(x_.ndim == 3 for x_ in x):
			return [np.stack(x, axis=0)]
		elif all(x_.ndim == 4 for x_ in x):
			return x
		elif any(x_.ndim > 4 for x_ in x):
			raise ValueError(f"Wrong dimension: x.ndim > 4.")

	raise ValueError(f"Wrong type: type(x)={type(x)}.")


def to_pil_image(image: Union[Tensor, np.ndarray]) -> PIL.Image:
	"""Convert image from `np.ndarray` or `Tensor` to PIL image."""
	if torch.is_tensor(image):
		# Equivalent to: `np_image = image.numpy()` but more efficient
		return torchvision.transforms.ToPILImage()(image)
	elif isinstance(image, np.ndarray):
		return PIL.Image.fromarray(image.astype(np.uint8), "RGB")

	raise TypeError(f"{type(image)} is not supported.")


@TRANSFORMS.register(name="to_image", force=True)
class ToImage(nn.Module):
	"""Converts a PyTorch tensor to a numpy image. In case the image is in the
	GPU, it will be copied back to CPU.

	Args:
		keep_dim (bool):
			If `False` squeeze the input image to match the shape [H, W, C] or
			[H, W]. Default: `True`.
		denormalize (bool):
			If `True`, converts the image in the range [0.0, 1.0] to the range
			[0, 255]. Default: `False`.
	"""

	def __init__(self, keep_dim: bool = True, denormalize: bool = False):
		super().__init__()
		self.keep_dim    = keep_dim
		self.denormalize = denormalize

	def forward(self, image: Tensor) -> np.ndarray:
		return to_image(image, self.keep_dim, self.denormalize)


@TRANSFORMS.register(name="tensor_list_to_image")
class TensorListToImage(nn.Module):
	"""Converts a list of tensors to a numpy image.

	Args:
		keep_dim (bool):
			If `False` squeeze the input image to match the shape [H, W, C] or
			[H, W]. Default: `True`.
		denormalize (bool):
			If `True`, converts the image in the range [0.0, 1.0] to the range
			[0, 255]. Default: `False`.
	"""

	def __init__(self, keep_dim: bool = True, denormalize: bool = False):
		super().__init__()
		self.keep_dim    = keep_dim
		self.denormalize = denormalize

	def forward(self, tensors: list[Tensor]) -> np.ndarray:
		return tensor_list_to_image(tensors, self.keep_dim, self.denormalize)
