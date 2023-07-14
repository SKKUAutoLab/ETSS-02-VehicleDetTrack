#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom data types.
"""

from __future__ import annotations

import functools
import types
from typing import Any
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union

import numpy as np
from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric

# MARK: - Templates
# Template for arguments which can be supplied as a tuple, or which can be a
# scalar which PyTorch will internally broadcast to a tuple. Comes in several
# variants: A tuple of unknown size, and a fixed-size tuple for 1d, 2d, or 3d
# operations.
T                      = TypeVar("T")
ScalarOrTuple1T        = Union[T, tuple[T]]
ScalarOrTuple2T        = Union[T, tuple[T, T]]
ScalarOrTuple3T        = Union[T, tuple[T, T, T]]
ScalarOrTuple4T        = Union[T, tuple[T, T, T, T]]
ScalarOrTuple5T        = Union[T, tuple[T, T, T, T, T]]
ScalarOrTuple6T        = Union[T, tuple[T, T, T, T, T, T]]
ScalarOrTupleAnyT      = Union[T, tuple[T, ...]]
ScalarListOrTupleAnyT  = Union[T, list[T], tuple[T, ...]]
ScalarOrCollectionAnyT = Union[T, list[T], tuple[T, ...], dict[Any, T]]
ListOrTupleAnyT        = Union[   list[T], tuple[T, ...]]
ListOrTuple2T          = Union[   list[T], tuple[T, T]]
ListOrTuple3T          = Union[   list[T], tuple[T, T, T]]
ListOrTuple4T          = Union[   list[T], tuple[T, T, T, T]]
ListOrTuple5T          = Union[   list[T], tuple[T, T, T, T, T]]
ListOrTuple6T          = Union[   list[T], tuple[T, T, T, T, T, T]]


# MARK: - Basic Types

Array1T     = ScalarOrTuple1T[np.ndarray]
Array2T     = ScalarOrTuple2T[np.ndarray]
Array3T     = ScalarOrTuple3T[np.ndarray]
Array4T     = ScalarOrTuple4T[np.ndarray]
Array5T     = ScalarOrTuple5T[np.ndarray]
Array6T     = ScalarOrTuple6T[np.ndarray]
ArrayAnyT   = ScalarOrTupleAnyT[np.ndarray]
ArrayList   = list[np.ndarray]
Arrays      = ScalarOrCollectionAnyT[np.ndarray]

Callable    = Union[str, type, object, types.FunctionType, functools.partial]
Color       = ListOrTuple3T[int]
Devices     = Union[ScalarListOrTupleAnyT[int], ScalarListOrTupleAnyT[str]]
Dim2        = ListOrTuple2T[int]
Dim3        = ListOrTuple3T[int]
Indexes     = ScalarListOrTupleAnyT[int]
Number      = Union[int, float]

Tensor1T    = ScalarOrTuple1T[Tensor]
Tensor2T    = ScalarOrTuple2T[Tensor]
Tensor3T    = ScalarOrTuple3T[Tensor]
Tensor4T    = ScalarOrTuple4T[Tensor]
Tensor5T    = ScalarOrTuple5T[Tensor]
Tensor6T    = ScalarOrTuple6T[Tensor]
TensorAnyT  = ScalarOrTupleAnyT[Tensor]
TensorList  = list[Tensor]
Tensors     = ScalarOrCollectionAnyT[Tensor]

Weights     = Union[Tensor, ListOrTupleAnyT[float], ListOrTupleAnyT[int]]


# MARK: - Layer's Parameters

Padding1T   = Union[ScalarOrTuple1T[int],   str]
Padding2T   = Union[ScalarOrTuple2T[int],   str]
Padding3T   = Union[ScalarOrTuple3T[int],   str]
Padding4T   = Union[ScalarOrTuple4T[int],   str]
Padding5T   = Union[ScalarOrTuple5T[int],   str]
Padding6T   = Union[ScalarOrTuple6T[int],   str]
PaddingAnyT = Union[ScalarOrTupleAnyT[int], str]

Size1T      = ScalarOrTuple1T[int]
Size2T      = ScalarOrTuple2T[int]
Size3T      = ScalarOrTuple3T[int]
Size4T      = ScalarOrTuple4T[int]
Size5T      = ScalarOrTuple5T[int]
Size6T      = ScalarOrTuple6T[int]
SizeAnyT    = ScalarOrTupleAnyT[int]


# MARK: - Model's Parameters

Config        = Union[str, dict, list]
Losses_       = Union[_Loss,     list[Union[_Loss,     dict]], dict]
Metrics_      = Union[Metric,    list[Union[Metric,    dict]], dict]
Optimizers_   = Union[Optimizer, list[Union[Optimizer, dict]], dict]

LabelTypes    = ScalarListOrTupleAnyT[str]
Metrics       = Union[dict[str, Tensor], dict[str, np.ndarray]]
Pretrained    = Union[bool, str, dict]
Tasks         = ScalarListOrTupleAnyT[str]

ForwardOutput = tuple[Tensors, Optional[Tensor]]
StepOutput    = Union[Tensor, dict[str, Any]]
EpochOutput   = list[StepOutput]
EvalOutput    = list[dict[str, float]]
PredictOutput = Union[list[Any], list[list[Any]]]


# MARK: - Data / Dataset / Datamodule

Augment_    = Union[dict, Callable]
Transform_  = Union[dict, Callable]
Transforms_ = Union[str, nn.Sequential, Transform_, list[Transform_]]

TrainDataLoaders = Union[
	DataLoader,
	Sequence[DataLoader],
	Sequence[Sequence[DataLoader]],
	Sequence[dict[str, DataLoader]],
	dict[str, DataLoader],
	dict[str, dict[str, DataLoader]],
	dict[str, Sequence[DataLoader]],
]
EvalDataLoaders = Union[DataLoader, Sequence[DataLoader]]
