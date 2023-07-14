# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Any
from typing import Optional

import torch
from packaging import version
from torch import Tensor

from core.utils import error_console


def torch_version() -> str:
	"""Parse the `torch.__version__` variable and removes +cu*/cpu."""
	return torch.__version__.split('+')[0]


def torch_version_geq(major, minor) -> bool:
	_version = version.parse(torch_version())
	return _version >= version.parse(f"{major}.{minor}")


if version.parse(torch_version()) > version.parse("1.7.1"):
	# TODO: remove the type: ignore once Python 3.6 is deprecated.
	# It turns out that Pytorch has no attribute `torch.linalg` for
	# Python 3.6 / PyTorch 1.7.0, 1.7.1
	from torch.linalg import solve  # type: ignore
else:
	from torch import solve as _solve

	# NOTE: in previous versions `torch.solve` accepted arguments in another order.
	def solve(A: Tensor, B: Tensor) -> Tensor:
		return _solve(B, A).solution


if version.parse(torch_version()) > version.parse("1.7.1"):
	# TODO: remove the type: ignore once Python 3.6 is deprecated.
	# It turns out that Pytorch has no attribute `torch.linalg` for
	# Python 3.6 / PyTorch 1.7.0, 1.7.1
	from torch.linalg import qr as linalg_qr  # type: ignore
else:
	from torch import qr as linalg_qr  # type: ignore # noqa: F401


def _extract_device_dtype(
		tensor_list: list[Optional[Any]]
) -> tuple[torch.device, torch.dtype]:
	"""Check if all the input are in the same device (only if when they are Tensor).

	If so, it would return a tuple of (device, dtype).
	Default: (cpu, `get_default_dtype()`).

	Returns:
		[torch.device, torch.dtype]
	"""
	device, dtype = None, None
	for tensor in tensor_list:
		if tensor is not None:
			if not isinstance(tensor, (Tensor,)):
				continue
			_device = tensor.device
			_dtype  = tensor.dtype
			if device is None and dtype is None:
				device = _device
				dtype  = _dtype
			elif device != _device or dtype != _dtype:
				raise ValueError(
					"Passed values are not in the same device and dtype."
					f"Got: ({device}, {dtype}) and ({_device}, {_dtype})."
				)
	if device is None:
		# TODO: update this when having torch.get_default_device()
		device = torch.device("cpu")
	if dtype is None:
		dtype = torch.get_default_dtype()
	return device, dtype


def _torch_inverse_cast(input: Tensor) -> Tensor:
	"""Helper function to make torch.inverse work with other than fp32/64.

	The function torch.inverse is only implemented for fp32/64 which makes
	impossible to be used by fp16 or others. What this function does, is cast
	input data type to fp32, apply torch.inverse, and cast back to the input
	dtype.
	"""
	if not isinstance(input, Tensor):
		raise AssertionError(f"Input must be Tensor. Got: {type(input)}.")
	dtype = input.dtype
	if dtype not in (torch.float32, torch.float64):
		dtype = torch.float32
	return torch.inverse(input.to(dtype)).to(input.dtype)


def _torch_histc_cast(input: Tensor, bins: int, min: int, max: int) -> Tensor:
	"""Helper function to make torch.histc work with other than fp32/64.

	The function torch.histc is only implemented for fp32/64 which makes
	impossible to be used by fp16 or others. What this function does, is cast
	input data type to fp32, apply torch.inverse, and cast back to the input
	dtype.
	"""
	if not isinstance(input, Tensor):
		raise AssertionError(f"Input must be Tensor. Got: {type(input)}.")
	dtype = input.dtype
	if dtype not in (torch.float32, torch.float64):
		dtype = torch.float32
	return torch.histc(input.to(dtype), bins, min, max).to(input.dtype)


def _torch_svd_cast(input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
	"""Helper function to make torch.svd work with other than fp32/64.

	The function torch.svd is only implemented for fp32/64 which makes
	impossible to be used by fp16 or others. What this function does, is cast
	input data type to fp32, apply torch.svd, and cast back to the input dtype.

	NOTE: in torch 1.8.1 this function is recommended to use as torch.linalg.svd
	"""
	if not isinstance(input, Tensor):
		raise AssertionError(f"Input must be Tensor. Got: {type(input)}.")
	dtype = input.dtype
	if dtype not in (torch.float32, torch.float64):
		dtype = torch.float32

	out1, out2, out3 = torch.svd(input.to(dtype))

	return out1.to(input.dtype), out2.to(input.dtype), out3.to(input.dtype)


# TODO: return only `Tensor` and review all the calls to adjust
def _torch_solve_cast(input: Tensor, A: Tensor) -> tuple[Tensor, Tensor]:
	"""Helper function to make torch.solve work with other than fp32/64.

	The function torch.solve is only implemented for fp32/64 which makes
	impossible to be used by fp16 or others. What this function does, is cast
	input data type to fp32, apply torch.svd, and cast back to the input dtype.
	"""
	if not isinstance(input, Tensor):
		raise AssertionError(f"Input must be Tensor. Got: {type(input)}.")
	dtype = input.dtype
	if dtype not in (torch.float32, torch.float64):
		dtype = torch.float32

	out = solve(A.to(dtype), input.to(dtype))

	return out.to(input.dtype), out


def safe_solve_with_mask(B: Tensor, A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
	r"""Helper function, which avoids crashing because of singular matrix input
	and outputs the mask of valid solution
	"""
	if not torch_version_geq(1, 10):
		sol, lu = _torch_solve_cast(B, A)
		error_console.log("PyTorch version < 1.10, solve validness mask maybe not "
						  "correct", RuntimeWarning)
		return sol, lu, torch.ones(len(A), dtype=torch.bool, device=A.device)
	# Based on https://github.com/pytorch/pytorch/issues/31546#issuecomment-694135622
	if not isinstance(B, Tensor):
		raise AssertionError(f"B must be Tensor. Got: {type(B)}.")
	dtype = B.dtype
	if dtype not in (torch.float32, torch.float64):
		dtype = torch.float32
	A_LU, pivots, info = torch.lu(A.to(dtype), get_infos=True)
	valid_mask         = info == 0
	X = torch.lu_solve(B.to(dtype), A_LU, pivots)
	return X.to(B.dtype), A_LU.to(A.dtype), valid_mask


def safe_inverse_with_mask(A: Tensor) -> tuple[Tensor, Tensor]:
	r"""Helper function, which avoids crashing because of non-invertable matrix
	input and outputs the mask of valid solution
	"""
	# Based on https://github.com/pytorch/pytorch/issues/31546#issuecomment-694135622
	if not torch_version_geq(1, 9):
		inv = _torch_inverse_cast(A)
		error_console.log("PyTorch version < 1.9, inverse validness mask maybe not "
						  "correct", RuntimeWarning)
		return inv, torch.ones(len(A), dtype=torch.bool, device=A.device)
	if not isinstance(A, Tensor):
		raise AssertionError(f"A must be Tensor. Got: {type(A)}.")
	dtype_original = A.dtype
	if dtype_original not in (torch.float32, torch.float64):
		dtype = torch.float32
	else:
		dtype = dtype_original
	from torch.linalg import inv_ex  # type: ignore # (not available in 1.8.1)
	inverse, info = inv_ex(A.to(dtype))
	mask          = info == 0
	return inverse.to(dtype_original), mask
