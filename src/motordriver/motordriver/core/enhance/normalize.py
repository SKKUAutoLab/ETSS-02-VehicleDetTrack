# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing functions for intensity normalisation.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn
from multipledispatch import dispatch
from torch import Tensor
from torchvision.transforms import Normalize
from torchvision.transforms.functional import normalize

from core.factory.builder import TRANSFORMS

__all__ = [
    "denormalize",
    "Denormalize",
    "denormalize_naive",
    "normalize",
    "Normalize",
    "normalize_min_max",
    "normalize_naive"
]


# MARK: - Normalize

def normalize_min_max(
    image  : Tensor,
    min_val: float = 0.0,
    max_val: float = 1.0,
    eps    : float = 1e-6
) -> Tensor:
    """Normalise an image/video image by MinMax and re-scales the value
    between a range.

    Args:
        image (Tensor):
            Ftensor image to be normalised with shape [B, C, *].
        min_val (float):
            Minimum value for the new range.
        max_val (float):
            Maximum value for the new range.
        eps (float):
            Float number to avoid zero division.

    Returns:
        x_out (Tensor):
            Fnormalized tensor image with same shape as input [B, C, *].

    Example:
        >>> x      = torch.rand(1, 5, 3, 3)
        >>> x_norm = normalize_min_max(image, min_val=-1., max_val=1.)
        >>> x_norm.min()
        image(-1.)
        >>> x_norm.max()
        image(1.0000)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"data should be a image. Got: {type(image)}.")
    if not isinstance(min_val, float):
        raise TypeError(f"'min_val' should be a float. Got: {type(min_val)}.")
    if not isinstance(max_val, float):
        raise TypeError(f"'b' should be a float. Got: {type(max_val)}.")
    if len(image.shape) < 3:
        raise ValueError(f"Input shape must be at least a 3d image. "
                         f"Got: {image.shape}.")

    shape = image.shape
    B, C  = shape[0], shape[1]

    x_min = image.view(B, C, -1).min(-1)[0].view(B, C, 1)
    x_max = image.view(B, C, -1).max(-1)[0].view(B, C, 1)

    x_out = ((max_val - min_val) * (image.view(B, C, -1) - x_min) /
             (x_max - x_min + eps) + min_val)
    return x_out.view(shape)


@dispatch(Tensor)
def normalize_naive(image: Tensor) -> Tensor:
    """Convert image from `torch.uint8` type and range [0, 255] to
    `torch.float` type and range of [0.0, 1.0].
    """
    default_float_dtype = torch.get_default_dtype()
    _image              = image.to(default_float_dtype)
    if abs(torch.max(_image)) > 1.0:
        _image = _image.div(255.0)
    return _image


@dispatch(np.ndarray)
def normalize_naive(image: np.ndarray) -> np.ndarray:
    """Convert image from `np.uint8` type and range [0, 255] to `np.float32`
    type and range of [0.0, 1.0].
    """
    _image = image.astype(np.float32)
    if abs(np.amax(_image)):
        _image /= 255.0
    return _image


@dispatch(list)
def normalize_naive(image: list) -> list:
    # NOTE: List of np.ndarray
    if all(isinstance(i, np.ndarray) and i.ndim == 3 for i in image):
        image = normalize_naive(np.array(image))
        return list(image)
    if all(isinstance(i, np.ndarray) and i.ndim == 4 for i in image):
        image = [normalize_naive(i) for i in image]
        return image
    
    # NOTE: List of Tensor
    if all(isinstance(i, Tensor) and i.ndim == 3 for i in image):
        image = normalize_naive(torch.stack(image))
        return list(image)
    if all(isinstance(i, Tensor) and i.ndim == 4 for i in image):
        image = [normalize_naive(i) for i in image]
        return image
    
    raise TypeError(f"Cannot normalize images of type: {type(image)}.")


@dispatch(tuple)
def normalize_naive(image: tuple) -> tuple:
    image = list(image)
    image = normalize_naive(image)
    return tuple(image)


@dispatch(dict)
def normalize_naive(image: dict) -> dict:
    if not all(
        isinstance(v, (tuple, list, Tensor, np.ndarray))
        for k, v in image.items()
    ):
        raise ValueError()
    
    for k, v in image.items():
        image[k] = normalize_naive(v)
    
    return image


# MARK: - Denormalize

def denormalize(
    data: Tensor,
    mean: Union[Tensor, float],
    std : Union[Tensor, float]
) -> Tensor:
    """Denormalize an image/video image with mean and standard deviation.
    
    input[channel] = (input[channel] * std[channel]) + mean[channel]
        
        where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n`
        channels,

    Args:
        data (Tensor):
            Tensor image of size [B, C, *].
        mean (Tensor, float):
            Mean for each channel.
        std (Tensor, float):
            Standard deviations for each channel.

    Return:
        out (Tensor):
            Denormalized image with same size as input [B, C, *].

    Examples:
        >>> x   = torch.rand(1, 4, 3, 3)
        >>> out = denormalize(x, 0.0, 255.)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x    = torch.rand(1, 4, 3, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std  = 255. * torch.ones(1, 4)
        >>> out  = denormalize(x, mean, std)
        >>> out.shape
        torch.Size([1, 4, 3, 3, 3])
    """
    shape = data.shape

    if isinstance(mean, float):
        mean = torch.tensor([mean] * shape[1], device=data.device,
                            dtype=data.dtype)
    if isinstance(std, float):
        std  = torch.tensor([std] * shape[1], device=data.device,
                            dtype=data.dtype)
    if not isinstance(data, Tensor):
        raise TypeError(f"data should be a image. Got: {type(data)}")
    if not isinstance(mean, Tensor):
        raise TypeError(f"mean should be a image or a float. Got: {type(mean)}")
    if not isinstance(std, Tensor):
        raise TypeError(f"std should be a image or float. Got: {type(std)}")

    # Allow broadcast on channel dimension
    if mean.shape and mean.shape[0] != 1:
        if mean.shape[0] != data.shape[-3] and mean.shape[:2] != data.shape[:2]:
            raise ValueError(f"mean length and number of channels do not "
                             f"match. Got: {mean.shape} and {data.shape}.")

    # Allow broadcast on channel dimension
    if std.shape and std.shape[0] != 1:
        if std.shape[0] != data.shape[-3] and std.shape[:2] != data.shape[:2]:
            raise ValueError(f"std length and number of channels do not "
                             f"match. Got: {std.shape} and {data.shape}.")

    mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
    std  = torch.as_tensor(std,  device=data.device, dtype=data.dtype)

    if mean.shape:
        mean = mean[..., :, None]
    if std.shape:
        std  = std[..., :, None]

    out = (data.view(shape[0], shape[1], -1) * std) + mean
    return out.view(shape)


@dispatch(Tensor)
def denormalize_naive(image: Tensor) -> Tensor:
    _image = image
    _image = torch.mul(_image, 255)
    _image = torch.clamp(_image, min=0, max=255)
    return _image.to(torch.uint8)


@dispatch(np.ndarray)
def denormalize_naive(image: np.ndarray) -> np.ndarray:
    _image = image
    _image = np.clip(_image * 255.0, 0, 255.0)
    return _image.astype(np.uint8)


@dispatch(list)
def denormalize_naive(image: list) -> list:
    # NOTE: List of np.ndarray
    if all(isinstance(i, np.ndarray) and i.ndim == 3 for i in image):
        return list(denormalize_naive(np.array(image)))
    if all(isinstance(i, np.ndarray) and i.ndim == 4 for i in image):
        return [denormalize_naive(i) for i in image]
    
    # NOTE: List of Tensor
    if all(isinstance(i, Tensor) and i.ndim == 3 for i in image):
        return list(denormalize_naive(torch.stack(image)))
    if all(isinstance(i, Tensor) and i.ndim == 4 for i in image):
        return [denormalize_naive(i) for i in image]
    
    raise TypeError(f"Cannot unnormalize images of type: {type(image)}.")


@dispatch(tuple)
def denormalize_naive(image: tuple) -> tuple:
    image = list(image)
    image = denormalize_naive(image)
    return tuple(image)


@dispatch(dict)
def denormalize_naive(image: dict) -> dict:
    if not all(
        isinstance(v, (tuple, list, Tensor, np.ndarray))
        for k, v in image.items()
    ):
        raise ValueError()
    
    for k, v in image.items():
        image[k] = denormalize_naive(v)
    
    return image


@TRANSFORMS.register(name="denormalize")
class Denormalize(nn.Module):
    """Denormalize a tensor image with mean and standard deviation.
 
    Args:
        mean (Tensor, float):
            Mean for each channel.
        std (Tensor, float):
            Standard deviations for each channel.

    Shape:
        - Input:  Tensor image of size [*, C, ...].
        - Output: Denormalized image with same size as input [*, C, ...].

    Examples:
        >>> x   = torch.rand(1, 4, 3, 3)
        >>> out = Denormalize(0.0, 255.)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x    = torch.rand(1, 4, 3, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std  = 255. * torch.ones(1, 4)
        >>> out  = Denormalize(mean, std)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3, 3])
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        mean: Union[Tensor, float],
        std : Union[Tensor, float]
    ):
        super().__init__()
        self.mean = mean
        self.std  = std

    def __repr__(self):
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return denormalize(image, self.mean, self.std)
