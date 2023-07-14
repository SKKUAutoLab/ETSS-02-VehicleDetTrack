#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base factory class for creating and registering classes.
"""

from __future__ import annotations

import inspect
from typing import Optional

# noinspection PyUnresolvedReferences

from core.type.type import Callable
from core.utils.rich import console
from core.utils.rich import print_table

__all__ = [
	"Registry"
]


# MARK: - Registry

class Registry:
	"""Base registry class for registering classes.

	Attributes:
		name (str):
			Registry name.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self, name: str):
		self._name     = name
		self._registry = {}
	
	def __len__(self):
		return len(self._registry)
	
	def __contains__(self, key: str):
		return self.get(key) is not None
	
	def __repr__(self):
		format_str = self.__class__.__name__ \
					 + f"(name={self._name}, items={self._registry})"
		return format_str
	
	# MARK: Properties
	
	@property
	def name(self) -> str:
		"""Return the registry's name."""
		return self._name
	
	@property
	def registry(self) -> dict:
		"""Return the registry's dictionary."""
		return self._registry
	
	def get(self, key: str) -> Callable:
		"""Get the registry record of the given `key`."""
		if key in self._registry:
			return self._registry[key]
	
	# MARK: Register
	
	def register(
		self,
		name  : Optional[str] = None,
		module: Callable	  = None,
		force : bool          = False
	) -> callable:
		"""Register a module.

		A record will be added to `self._registry`, whose key is the class name
		or the specified name, and value is the class itself. It can be used
		as a decorator or a normal function.

		Example:
			# >>> backbones = Factory("backbone")
			# >>>
			# >>> @backbones.register()
			# >>> class ResNet:
			# >>>     pass
			# >>>
			# >>> @backbones.register(name="mnet")
			# >>> class MobileNet:
			# >>>     pass
			# >>>
			# >>> class ResNet:
			# >>>     pass
			# >>> backbones.register(ResNet)

		Args:
			name (str, optional):
				Module name to be registered. If not specified, the class
				name will be used.
			module (type):
				Module class to be registered.
			force (bool):
				Whether to override an existing class with the same name.
		"""
		if not (name is None or isinstance(name, str)):
			raise TypeError(
				f"`name` must be either of `None` or an instance of `str`, "
				f"but got {type(name)}."
			)
		
		# NOTE: Use it as a normal method: x.register(module=SomeClass)
		if module is not None:
			self.register_module(module, name, force)
			return module
		
		# NOTE: Use it as a decorator: @x.register()
		def _register(cls):
			self.register_module(cls, name, force)
			return cls
		
		return _register
	
	def register_module(
		self,
		module_class: Callable,
		module_name : Optional[str] = None,
		force	    : bool 			= False
	):
		if not inspect.isclass(module_class):
			raise TypeError(
				f"Module must be a class. But got: {type(module_class)}."
			)
		
		if module_name is None:
			module_name = module_class.__name__.lower()
		
		if isinstance(module_name, str):
			module_name = [module_name]
		
		for name in module_name:
			if not force and name in self._registry:
				continue
				# logger.debug(f"{name} is already registered in {self.name}.")
			else:
				self._registry[name] = module_class
	
	# MARK: Print

	def print(self):
		"""Print the registry dictionary."""
		console.log(f"[red]{self.name}:")
		print_table(self.registry)
