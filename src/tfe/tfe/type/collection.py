#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations on collections.
"""

from __future__ import annotations

import collections
import itertools
from collections import abc
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import torch
from multipledispatch import dispatch
from torch import Tensor

__all__ = [
	"concat_list",
	"eye_like",
	"is_dict_of",
	"is_list_of",
	"is_seq_of",
	"is_tuple_of",
	"slice_list",
	"to_1tuple",
	"to_2tuple",
	"to_3tuple",
	"to_4tuple",
	"to_3d_array",
	"to_3d_tensor",
	"to_4d_array",
	"to_4d_array_list",
	"to_4d_tensor",
	"to_4d_tensor_list",
	"to_5d_array",
	"to_5d_tensor",
	"to_iter",
	"to_list",
	"to_ntuple",
	"to_tuple",
	"unique",
	"vec_like",
]


# MARK: - Construction

def eye_like(n: int, input: Tensor) -> Tensor:
	r"""Return a 2-D image with ones on the diagonal and zeros elsewhere with
	the same batch size as the input.

	Args:
		n (int):
			Number of rows [N].
		input (Tensor):
			Tensor that will determine the batch size of the output matrix.
			The expected shape is [B, *].

	Returns:
		(Tensor):
			The identity matrix with the same batch size as the input [B, N, N].

	"""
	if n <= 0:
		raise AssertionError(type(n), n)
	if len(input.shape) < 1:
		raise AssertionError(input.shape)

	identity = torch.eye(n, device=input.device, dtype=input.dtype)
	return identity[None].repeat(input.shape[0], 1, 1)


def vec_like(n: int, input: Tensor):
	r"""Return a 2-D image with a vector containing zeros with the same batch
	size as the input.

	Args:
		n (int):
			Number of rows [N].
		input (Tensor):
			Tensor that will determine the batch size of the output matrix.
			The expected shape is [B, *].

	Returns:
		(Tensor):
			The vector with the same batch size as the input [B, N, 1].

	"""
	if n <= 0:
		raise AssertionError(type(n), n)
	if len(input.shape) < 1:
		raise AssertionError(input.shape)

	vec = torch.zeros(n, 1, device=input.device, dtype=input.dtype)
	return vec[None].repeat(input.shape[0], 1, 1)


# MARK: - Conversion

def to_iter(
		inputs: Iterable, dst_type: type, return_type: Optional[type] = None
):
	"""Cast elements of an iterable object into some type.

	Args:
		inputs (Iterable):
			Input object.
		dst_type (type):
			Destination type.
		return_type (type, optional):
			If specified, the output object will be converted to this type,
			otherwise an iterator.
	"""
	if not isinstance(inputs, abc.Iterable):
		raise TypeError("`inputs` must be an iterable object.")
	if not isinstance(dst_type, type):
		raise TypeError("`dst_type` must be a valid type.")

	out_iterable = map(dst_type, inputs)

	if return_type is None:
		return out_iterable
	else:
		return return_type(out_iterable)


def to_list(inputs: Iterable, dst_type: type):
	"""Cast elements of an iterable object into a list of some type. A partial
	method of `to_iter()`.
	"""
	return to_iter(inputs=inputs, dst_type=dst_type, return_type=list)


def to_tuple(inputs: Iterable, dst_type: type):
	"""Cast elements of an iterable object into a tuple of some type. A partial
	method of `to_iter()`."""
	return to_iter(inputs=inputs, dst_type=dst_type, return_type=tuple)


def to_ntuple(n: int):
	"""A helper functions to cast input to n-tuple."""
	def parse(x) -> tuple:
		if isinstance(x, collections.abc.Iterable):
			return tuple(x)
		return tuple(itertools.repeat(x, n))
	return parse


to_1tuple = to_ntuple(1)
to_2tuple = to_ntuple(2)
to_3tuple = to_ntuple(3)
to_4tuple = to_ntuple(4)


def to_3d_array(input) -> np.ndarray:
	"""Convert to a 3D array."""
	if isinstance(input, Tensor):
		input = input.detach().cpu().numpy()

	if isinstance(input, np.ndarray):
		if input.ndim < 3:
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif input.ndim == 4 and input.shape[0] == 1:
			input = np.squeeze(input, axis=0)
		elif input.ndim > 4:
			raise ValueError(f"Wrong dimension: input.ndim > 4.")
		return input

	raise ValueError(f"Wrong type: type(input)={type(input)}.")


def to_4d_array(input) -> np.ndarray:
	"""Convert to a 4D-array. The output will be:
		- Single 3D-array will be expanded to a single 4D-array.
		- Single 4D-array will remain the same.
		- Sequence of 3D-arrays will be stacked into a 4D-array.
		- Sequence of 4D-arrays will remain the same.
	"""
	if isinstance(input, Tensor):
		input = input.detach().cpu().numpy()

	if isinstance(input, np.ndarray):
		if input.ndim < 3:
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif input.ndim == 3:
			input = np.expand_dims(input, axis=0)
		elif input.ndim > 4:
			raise ValueError(f"Wrong dimension: input.ndim > 4.")
		return input

	if isinstance(input, tuple):
		input = list(input)

	if isinstance(input, dict):
		input = [v for k, v in input.items()]

	if isinstance(input, list) and is_list_of(input, Tensor):
		input = [_x.detach().cpu().numpy() for _x in input]

	if isinstance(input, list) and is_list_of(input, np.ndarray):
		if any(x_.ndim < 3 for x_ in input):
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif all(x_.ndim == 3 for x_ in input):
			return to_4d_array(input=np.stack(input))
		elif any(x_.ndim > 3 for x_ in input):
			raise ValueError(f"Wrong dimension: input.ndim > 4.")

	raise ValueError(f"Wrong type: type(input)={type(input)}.")


def to_5d_array(input) -> np.ndarray:
	"""Convert to a 5D-array."""
	if isinstance(input, Tensor):
		input = input.detach().cpu().numpy()

	if isinstance(input, np.ndarray):
		if input.ndim < 3:
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif input.ndim == 3:
			input = np.expand_dims(input, axis=0)
			input = np.expand_dims(input, axis=0)
		elif input.ndim == 4:
			input = np.expand_dims(input, axis=0)
		elif input.ndim > 4:
			raise ValueError(f"Wrong dimension: input.ndim > 4.")
		return input

	if isinstance(input, tuple):
		input = list(input)

	if isinstance(input, dict):
		input = [v for k, v in input.items()]

	if isinstance(input, list) and is_list_of(input, Tensor):
		input = [_x.detach().cpu().numpy() for _x in input]

	if isinstance(input, list) and is_list_of(input, np.ndarray):
		if any(x_.ndim < 3 for x_ in input):
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif all(3 <= x_.ndim <= 4 for x_ in input):
			return to_5d_array(input=np.stack(input))
		elif any(x_.ndim > 4 for x_ in input):
			raise ValueError(f"Wrong dimension: input.ndim > 4.")

	raise ValueError(f"Wrong type: type(input)={type(input)}.")


def to_4d_array_list(input) -> list[np.ndarray]:
	"""Convert to a 4D-array list."""
	if isinstance(input, Tensor):
		input = input.detach().cpu().numpy()

	if isinstance(input, np.ndarray):
		if input.ndim < 3:
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif input.ndim == 3:
			input = [np.expand_dims(input, axis=0)]
		elif input.ndim == 4:
			input = [input]
		elif input.ndim == 5:
			input = list(input)
		return input

	if isinstance(input, tuple):
		input = list(input)

	if isinstance(input, dict):
		input = [v for k, v in input.items()]

	if isinstance(input, list) and is_list_of(input, Tensor):
		input = [_x.detach().cpu().numpy() for _x in input]

	if isinstance(input, list) and is_list_of(input, np.ndarray):
		if all(x_.ndim < 3 for x_ in input):
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif all(x_.ndim == 3 for x_ in input):
			return [np.stack(input, axis=0)]
		elif all(x_.ndim == 4 for x_ in input):
			return input
		elif any(x_.ndim > 4 for x_ in input):
			raise ValueError(f"Wrong dimension: input.ndim > 4.")

	raise ValueError(f"Wrong type: type(input)={type(input)}.")


def to_3d_tensor(input) -> Tensor:
	"""Convert to a 3D tensor."""
	if isinstance(input, np.ndarray):
		input = torch.from_numpy(input)

	if isinstance(input, Tensor):
		if input.ndim < 3:
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif input.ndim == 4 and input.shape[0] == 1:
			input = input.unsqueeze(dim=0)
		elif input.ndim > 4:
			raise ValueError(f"Wrong dimension: input.ndim > 4.")
		return input

	raise ValueError(f"Wrong type: type(input)={type(input)}.")


def to_4d_tensor(input) -> Tensor:
	"""Convert to a 4D tensor. The output will be:
		- Single 3D tensor will be expanded to a 4D tensor.
		- Single 4D tensor will remain the same.
		- Sequence of 3D tensors will be stacked into a 4D tensor.
		- Sequence of 4D tensors will remain the same.
	"""
	if isinstance(input, np.ndarray):
		input = torch.from_numpy(input)

	if isinstance(input, Tensor):
		if input.ndim < 3:
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif input.ndim == 3:
			input = input.unsqueeze(dim=0)
		elif input.ndim > 4:
			raise ValueError(f"Wrong dimension: input.ndim > 4.")
		return input

	if isinstance(input, tuple):
		input = list(input)

	if isinstance(input, dict):
		input = [v for k, v in input.items()]

	if isinstance(input, list) and is_list_of(input, np.ndarray):
		input = [torch.from_numpy(_x) for _x in input]

	if isinstance(input, list) and is_list_of(input, Tensor):
		if any(x_.ndim < 3 for x_ in input):
			raise ValueError("Wrong dimension: input.ndim < 3.")
		elif all(x_.ndim == 3 for x_ in input):
			return to_4d_tensor(input=torch.stack(input, dim=0))
		elif any(x_.ndim > 3 for x_ in input):
			raise ValueError("Wrong dimension: input.ndim > 3.")

	raise ValueError(f"Wrong type: type(input)={type(input)}.")


def to_5d_tensor(input) -> Tensor:
	"""Convert to a 5D tensor."""
	if isinstance(input, np.ndarray):
		input = torch.from_numpy(input)

	if isinstance(input, Tensor):
		if input.ndim < 3:
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif input.ndim == 3:
			input = input.unsqueeze(dim=0)
			input = input.unsqueeze(dim=0)
		elif input.ndim == 4:
			input = input.unsqueeze(dim=0)
		elif input.ndim > 5:
			raise ValueError(f"Wrong dimension: input.ndim > 5.")
		return input

	if isinstance(input, tuple):
		input = list(input)

	if isinstance(input, dict):
		input = [v for k, v in input.items()]

	if isinstance(input, list) and is_list_of(input, np.ndarray):
		input = [torch.from_numpy(_x) for _x in input]

	if isinstance(input, list) and is_list_of(input, Tensor):
		if any(x_.ndim < 3 for x_ in input):
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif all(3 <= x_.ndim <= 4 for x_ in input):
			return to_5d_tensor(input=torch.stack(input, dim=0))
		elif any(x_.ndim > 4 for x_ in input):
			raise ValueError(f"Wrong dimension: input.ndim > 4.")

	raise ValueError(f"Wrong type: type(input)={type(input)}.")


def to_4d_tensor_list(input) -> list[Tensor]:
	"""Convert to a list of 4D tensors."""
	if isinstance(input, np.ndarray):
		input = torch.from_numpy(input)

	if isinstance(input, Tensor):
		if input.ndim < 3:
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif input.dim() == 3:
			input = [input.unsqueeze(dim=0)]
		elif input.dim() == 4:
			input = [input]
		elif input.dim() == 5:
			input = list(input)
		elif input.ndim > 5:
			raise ValueError(f"Wrong dimension: input.ndim > 5.")

	if isinstance(input, tuple):
		input = list(input)

	if isinstance(input, dict):
		input = [v for k, v in input.items()]
		return to_4d_tensor_list(input=input)

	if isinstance(input, list) and is_list_of(input, np.ndarray):
		input = [torch.from_numpy(_x) for _x in input]

	if isinstance(input, list) and is_list_of(input, Tensor):
		if any(x_.ndim < 3 for x_ in input):
			raise ValueError(f"Wrong dimension: input.ndim < 3.")
		elif all(x_.ndim == 3 for x_ in input):
			return [torch.stack(input, dim=0)]
		elif all(x_.ndim == 4 for x_ in input):
			return input
		elif any(x_.ndim > 4 for x_ in input):
			raise ValueError(f"Wrong dimension: input.ndim > 4.")

	raise ValueError(f"Wrong type: type(input)={type(input)}.")


# MARK: - Modifying

def slice_list(in_list: list, lens: Union[int, list]) -> list[list]:
	"""Slice a list into several sub lists by a list of given length.

	Args:
		in_list (list):
			List to be sliced.
		lens(int, list):
			Expected length of each out list.

	Returns:
		out_list (list):
			A list of sliced list.
	"""
	if isinstance(lens, int):
		if len(in_list) % lens != 0:
			raise ValueError
		lens = [lens] * int(len(in_list) / lens)
	if not isinstance(lens, list):
		raise TypeError("`indices` must be an integer or a list of integers.")
	elif sum(lens) != len(in_list):
		raise ValueError(f"Sum of lens and list length does not match: "
						 f"{sum(lens)} != {len(in_list)}.")

	out_list = []
	idx      = 0
	for i in range(len(lens)):
		out_list.append(in_list[idx:idx + lens[i]])
		idx += lens[i]
	return out_list


def concat_list(in_list: list) -> list:
	"""Concatenate a list of list into a single list."""
	return list(itertools.chain(*in_list))


@dispatch(list)
def unique(in_list: list) -> list:
	"""Return a list with only unique elements."""
	return list(set(in_list))


@dispatch(tuple)
def unique(in_tuple: tuple) -> tuple:
	"""Return a tuple with only unique elements."""
	return tuple(set(in_tuple))


@dispatch(tuple)
def unique(in_tuple: tuple) -> tuple:
	"""Return a tuple with only unique elements."""
	return tuple(set(in_tuple))


# MARK: - Validation

def is_seq_of(
		seq: Sequence, expected_type: type, seq_type: Optional[type] = None
) -> bool:
	"""Check whether it is a sequence of some type.

	Args:
		seq (Sequence):
			Sequence to be checked.
		expected_type (type):
			Expected type of sequence items.
		seq_type (type, optional):
			Expected sequence type.
	"""
	if seq_type is None:
		exp_seq_type = abc.Sequence
	else:
		if not isinstance(seq_type, type):
			raise ValueError
		exp_seq_type = seq_type
	if not isinstance(seq, exp_seq_type):
		return False
	for item in seq:
		if not isinstance(item, expected_type):
			return False
	return True


def is_list_of(seq: list, expected_type: type) -> bool:
	"""Check whether it is a list of some type. A partial method of
	`is_seq_of()`.
	"""
	return is_seq_of(seq=seq, expected_type=expected_type, seq_type=list)


def is_tuple_of(seq: tuple, expected_type: type) -> bool:
	"""Check whether it is a tuple of some type. A partial method of
	`is_seq_of()`."""
	return is_seq_of(seq=seq, expected_type=expected_type, seq_type=tuple)


def is_dict_of(d: dict, expected_type: type) -> bool:
	"""Check whether it is a dict of some type."""
	if not isinstance(expected_type, type):
		assert ValueError
	return all(isinstance(v, expected_type) for k, v in d.items())
