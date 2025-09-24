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
from typing_extensions import TypeAlias as _TypeAlias

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
ScalarOrTuple1T        : _TypeAlias = Union[T, tuple[T]]
ScalarOrTuple2T        : _TypeAlias = Union[T, tuple[T, T]]
ScalarOrTuple3T        : _TypeAlias = Union[T, tuple[T, T, T]]
ScalarOrTuple4T        : _TypeAlias = Union[T, tuple[T, T, T, T]]
ScalarOrTuple5T        : _TypeAlias = Union[T, tuple[T, T, T, T, T]]
ScalarOrTuple6T        : _TypeAlias = Union[T, tuple[T, T, T, T, T, T]]
ScalarOrTupleAnyT      : _TypeAlias = Union[T, tuple[T, ...]]
ScalarListOrTupleAnyT  : _TypeAlias = Union[T, list[T], tuple[T, ...]]
ScalarOrCollectionAnyT : _TypeAlias = Union[T, list[T], tuple[T, ...], dict[Any, T]]
ListOrTupleAnyT        : _TypeAlias = Union[   list[T], tuple[T, ...]]
ListOrTuple2T          : _TypeAlias = Union[   list[T], tuple[T, T]]
ListOrTuple3T          : _TypeAlias = Union[   list[T], tuple[T, T, T]]
ListOrTuple4T          : _TypeAlias = Union[   list[T], tuple[T, T, T, T]]
ListOrTuple5T          : _TypeAlias = Union[   list[T], tuple[T, T, T, T, T]]
ListOrTuple6T          : _TypeAlias = Union[   list[T], tuple[T, T, T, T, T, T]]


# MARK: - Basic Types

Array1T     : _TypeAlias = ScalarOrTuple1T[np.ndarray]
Array2T     : _TypeAlias = ScalarOrTuple2T[np.ndarray]
Array3T     : _TypeAlias = ScalarOrTuple3T[np.ndarray]
Array4T     : _TypeAlias = ScalarOrTuple4T[np.ndarray]
Array5T     : _TypeAlias = ScalarOrTuple5T[np.ndarray]
Array6T     : _TypeAlias = ScalarOrTuple6T[np.ndarray]
ArrayAnyT   : _TypeAlias = ScalarOrTupleAnyT[np.ndarray]
ArrayList   : _TypeAlias = list[np.ndarray]
Arrays      : _TypeAlias = ScalarOrCollectionAnyT[np.ndarray]

Callable    : _TypeAlias = Union[str, type, object, types.FunctionType, functools.partial]
Color       : _TypeAlias = ListOrTuple3T[int]
Devices     : _TypeAlias = Union[ScalarListOrTupleAnyT[int], ScalarListOrTupleAnyT[str]]
Dim2        : _TypeAlias = ListOrTuple2T[int]
Dim3        : _TypeAlias = ListOrTuple3T[int]
Indexes     : _TypeAlias = ScalarListOrTupleAnyT[int]
Number      : _TypeAlias = Union[int, float]

Tensor1T    : _TypeAlias = ScalarOrTuple1T[Tensor]
Tensor2T    : _TypeAlias = ScalarOrTuple2T[Tensor]
Tensor3T    : _TypeAlias = ScalarOrTuple3T[Tensor]
Tensor4T    : _TypeAlias = ScalarOrTuple4T[Tensor]
Tensor5T    : _TypeAlias = ScalarOrTuple5T[Tensor]
Tensor6T    : _TypeAlias = ScalarOrTuple6T[Tensor]
TensorAnyT  : _TypeAlias = ScalarOrTupleAnyT[Tensor]
TensorList  : _TypeAlias = list[Tensor]
Tensors     : _TypeAlias = ScalarOrCollectionAnyT[Tensor]

Weights     : _TypeAlias = Union[Tensor, ListOrTupleAnyT[float], ListOrTupleAnyT[int]]


# MARK: - Layer's Parameters

Padding1T   : _TypeAlias = Union[ScalarOrTuple1T[int],   str]
Padding2T   : _TypeAlias = Union[ScalarOrTuple2T[int],   str]
Padding3T   : _TypeAlias = Union[ScalarOrTuple3T[int],   str]
Padding4T   : _TypeAlias = Union[ScalarOrTuple4T[int],   str]
Padding5T   : _TypeAlias = Union[ScalarOrTuple5T[int],   str]
Padding6T   : _TypeAlias = Union[ScalarOrTuple6T[int],   str]
PaddingAnyT : _TypeAlias = Union[ScalarOrTupleAnyT[int], str]

Size1T      : _TypeAlias = ScalarOrTuple1T[int]
Size2T      : _TypeAlias = ScalarOrTuple2T[int]
Size3T      : _TypeAlias = ScalarOrTuple3T[int]
Size4T      : _TypeAlias = ScalarOrTuple4T[int]
Size5T      : _TypeAlias = ScalarOrTuple5T[int]
Size6T      : _TypeAlias = ScalarOrTuple6T[int]
SizeAnyT    : _TypeAlias = ScalarOrTupleAnyT[int]


# MARK: - Model's Parameters

Config        : _TypeAlias = Union[str, dict, list]
Losses_       : _TypeAlias = Union[_Loss,     list[Union[_Loss,     dict]], dict]
Metrics_      : _TypeAlias = Union[Metric,    list[Union[Metric,    dict]], dict]
Optimizers_   : _TypeAlias = Union[Optimizer, list[Union[Optimizer, dict]], dict]

LabelTypes    : _TypeAlias = ScalarListOrTupleAnyT[str]
Metrics       : _TypeAlias = Union[dict[str, Tensor], dict[str, np.ndarray]]
Pretrained    : _TypeAlias = Union[bool, str, dict]
Tasks         : _TypeAlias = ScalarListOrTupleAnyT[str]

ForwardOutput : _TypeAlias = tuple[Tensors, Optional[Tensor]]
StepOutput    : _TypeAlias = Union[Tensor, dict[str, Any]]
EpochOutput   : _TypeAlias = list[StepOutput]
EvalOutput    : _TypeAlias = list[dict[str, float]]
PredictOutput : _TypeAlias = Union[list[Any], list[list[Any]]]


# MARK: - Data / Dataset / Datamodule

Augment_    : _TypeAlias = Union[dict, Callable]
Transform_  : _TypeAlias = Union[dict, Callable]
Transforms_ : _TypeAlias = Union[str, nn.Sequential, Transform_, list[Transform_]]

TrainDataLoaders : _TypeAlias = Union[
	DataLoader,
	Sequence[DataLoader],
	Sequence[Sequence[DataLoader]],
	Sequence[dict[str, DataLoader]],
	dict[str, DataLoader],
	dict[str, dict[str, DataLoader]],
	dict[str, Sequence[DataLoader]],
]
EvalDataLoaders : _TypeAlias = Union[DataLoader, Sequence[DataLoader]]
