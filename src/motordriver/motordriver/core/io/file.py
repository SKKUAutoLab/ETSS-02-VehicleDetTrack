#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Function for parsing and dumping data to several file format, such as:
yaml, txt, json, ...
"""

from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod

import pickle
import yaml
import json

from pathlib import Path
from typing import Any
from typing import Optional
from typing import TextIO
from typing import Union
import numpy as np

from core.factory.builder import FILE_HANDLERS

__all__ = [
	"dump",
	"load",
	"BaseFileHandler",
	"PickleHandler",
	"JsonHandler"
]

# MARK: - Load

def load(
		path: Union[str, Path, TextIO], file_format: Optional[str] = None, **kwargs
) -> Union[str, dict, None]:
	"""Load data from json/yaml/pickle files. This method provides a unified
	api for loading data from serialized files.

	Args:
		path (str, Path, TextIO):
			Filename, path, or a file-like object.
		file_format (str, optional):
			If not specified, the file format will be inferred from the file
			extension, otherwise use the specified one. Currently supported
			formats include "json", "yaml/yml" and "pickle/pkl".

	Returns:
		data (str, dict, optional):
			Content from the file.
	"""
	if isinstance(path, Path):
		path = str(path)
	if file_format is None and isinstance(path, str):
		file_format = path.split(".")[-1]
	if file_format not in FILE_HANDLERS:
		raise TypeError(f"Unsupported format: {file_format}.")

	handler = FILE_HANDLERS.build(name=file_format)
	if isinstance(path, str):
		data = handler.load_from_file(path, **kwargs)
	elif hasattr(path, "read"):
		data = handler.load_from_fileobj(path, **kwargs)
	else:
		raise TypeError("`file` must be a filepath str or a file-object.")
	return data


# MARK: - Dump

def dump(
		obj: Any,
		path: Union[str, Path, TextIO],
		file_format: Optional[str] = None,
		**kwargs
) -> Union[bool, str]:
	"""Dump data to json/yaml/pickle strings or files. This method provides a
	unified api for dumping data as strings or to files, and also supports
	custom arguments for each file format.

	Args:
		obj (any):
			Python object to be dumped.
		path (str, Path, TextIO):
			If not specified, then the object is dump to a str, otherwise to a
			file specified by the filename or file-like object.
		file_format (str, optional):
			If not specified, the file format will be inferred from the file
			extension, otherwise use the specified one. Currently supported
			formats include "json", "yaml/yml" and "pickle/pkl".

	Returns:
		(bool, str):
			`True` for success, `False` otherwise.
	"""
	if isinstance(path, Path):
		path = str(path)
	if file_format is None:
		if isinstance(path, str):
			file_format = path.split(".")[-1]
		elif path is None:
			raise ValueError(
				"`file_format` must be specified since file is None."
			)
	if file_format not in FILE_HANDLERS:
		raise TypeError(f"Unsupported format: {file_format}.")

	handler = FILE_HANDLERS.build(name=file_format)
	if path is None:
		return handler.dump_to_str(obj, **kwargs)
	elif isinstance(path, str):
		handler.dump_to_file(obj, path, **kwargs)
	elif hasattr(path, "write"):
		handler.dump_to_fileobj(obj, path, **kwargs)
	else:
		raise TypeError("`file` must be a filename str or a file-object.")


# MARK: - BaseFileHandler

class BaseFileHandler(metaclass=ABCMeta):
	"""Base file handler implements the template methods (i.e., skeleton) for
	read and write data from/to different file formats.
	"""

	@abstractmethod
	def load_from_fileobj(
			self, path: Union[str, TextIO], **kwargs
	) -> Optional[Union[str, dict]]:
		"""Load the content from the given filepath or file-like object
		(input stream).
		"""
		pass

	@abstractmethod
	def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
		"""Dump data from the given obj to the filepath or file-like object.
		"""
		pass

	@abstractmethod
	def dump_to_str(self, obj, **kwargs) -> str:
		"""Dump data from the given obj to string."""
		pass

	def load_from_file(
			self, path: str, mode: str = "r", **kwargs
	) -> Optional[Union[str, dict]]:
		"""Load content from the given file."""
		with open(path, mode) as f:
			return self.load_from_fileobj(f, **kwargs)

	def dump_to_file(self, obj, path: str, mode: str = "w", **kwargs):
		"""Dump data from object to file.

		Args:
			obj:
				Object.
			path (str):
				Filepath.
			mode (str):
				File opening mode.
		"""
		with open(path, mode) as f:
			self.dump_to_fileobj(obj, f, **kwargs)


# MARK: - PickleHandler

@FILE_HANDLERS.register(name="pickle")
@FILE_HANDLERS.register(name="pkl")
class PickleHandler(BaseFileHandler):
	"""Pickle file handler."""

	def load_from_fileobj(
			self, path: Union[str, TextIO], **kwargs
	) -> Optional[Union[str, dict]]:
		"""Load the content from the given filepath or file-like object
		(input stream).
		"""
		return pickle.load(path, **kwargs)

	def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
		"""Dump data from the given obj to the filepath or file-like object.
		"""
		kwargs.setdefault("protocol", 2)
		pickle.dump(obj, path, **kwargs)

	def dump_to_str(self, obj, **kwargs) -> bytes:
		""""Dump data from the given obj to string."""
		kwargs.setdefault("protocol", 2)
		return pickle.dumps(obj, **kwargs)

	def load_from_file(
			self, file: Union[str, Path], **kwargs
	) -> Optional[Union[str, dict]]:
		"""Load content from the given file."""
		return super().load_from_file(file, mode="rb", **kwargs)

	def dump_to_file(self, obj, path: Union[str, Path], **kwargs):
		"""Dump data from object to file.

		Args:
			obj:
				Object.
			path (str, Path):
				Filepath.
		"""
		super().dump_to_file(obj, path, mode="wb", **kwargs)

# MARK: - YamlHandler

try:
	from yaml import CLoader as FullLoader, CDumper as Dumper
except ImportError:
	from yaml import FullLoader, Dumper

__all__ = [
	"YamlHandler"
]

@FILE_HANDLERS.register(name="yaml")
@FILE_HANDLERS.register(name="yml")
class YamlHandler(BaseFileHandler):
	"""YAML file handler."""

	def load_from_fileobj(
			self, path: Union[str, TextIO], **kwargs
	) -> Optional[Union[str, dict]]:
		"""Load the content from the given filepath or file-like object
		(input stream).
		"""
		kwargs.setdefault("Loader", FullLoader)
		return yaml.load(path, **kwargs)

	def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
		"""Dump data from the given obj to the filepath or file-like object.
		"""
		kwargs.setdefault("Dumper", Dumper)
		yaml.dump(obj, path, **kwargs)

	def dump_to_str(self, obj, **kwargs) -> str:
		"""Dump data from the given obj to string."""
		kwargs.setdefault("Dumper", Dumper)
		return yaml.dump(obj, **kwargs)


# MARK: - JsonHandler

@FILE_HANDLERS.register(name="json")
class JsonHandler(BaseFileHandler):
	"""JSON file handler."""

	@staticmethod
	def set_default(obj):
		"""Set default json values for non-serializable values. It helps
		convert `set`, `range` and `np.ndarray` data types to list. It also
		converts `np.generic` (including `np.int32`, `np.float32`, etc.) into
		plain numbers of plain python built-in types.
		"""
		if isinstance(obj, (set, range)):
			return list(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, np.generic):
			return obj.item()
		raise TypeError(f"{type(obj)} is unsupported for json dump")

	def load_from_fileobj(
			self, path: Union[str, TextIO], **kwargs
	) -> Optional[Union[str, dict]]:
		"""Load the content from the given filepath or file-like object
		(input stream).
		"""
		return json.load(path)

	def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
		"""Dump data from the given obj to the filepath or file-like object.
		"""
		kwargs.setdefault("default", self.set_default)
		json.dump(obj, path, **kwargs)

	def dump_to_str(self, obj, **kwargs) -> str:
		"""Dump data from the given obj to string."""
		kwargs.setdefault("default", self.set_default)
		return json.dumps(obj, **kwargs)
