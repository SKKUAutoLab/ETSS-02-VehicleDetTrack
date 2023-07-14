#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base factory class for creating and registering classes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional
from typing import Union

from munch import Munch

from core.type.collection import is_list_of
from core.utils.rich import error_console
from core.factory.registry import Registry

__all__ = [
    "Factory",
]


# MARK: - Factory

class Factory(Registry):
	"""Default factory class for creating objects.
	
	Registered object could be built from registry.
    Example:
        >>> MODELS = Factory("models")
        >>> @MODELS.register()
        >>> class ResNet:
        >>>     pass
        >>>
        >>> resnet_hparams = {}
        >>> resnet         = MODELS.build(name="ResNet", **resnet_hparams)
	"""
	
	# MARK: Build
	
	def build(self, name: str, *args, **kwargs) -> object:
		"""Factory command to create an instance of the class. This method gets
		the appropriate class from the registry and creates an instance of
		it, while passing in the parameters given in `kwargs`.
		
		Args:
			name (str):
				Name of the class to create.
			
		Returns:
			instance (object, optional):
				An instance of the class that is created.
		"""
		if name not in self.registry:
			error_console.log(f"{name} does not exist in the registry.")
			return None
		
		instance = self.registry[name](*args, **kwargs)
		if not hasattr(instance, "name"):
			instance.name = name
		return instance
	
	def build_from_dict(
		self, cfg: Optional[Union[dict, Munch]], **kwargs
	) -> Optional[object]:
		"""Factory command to create an instance of a class. This method gets
		the appropriate class from the registry while passing in the
		parameters given in `cfg`.
		
		Args:
			cfg (dict, Munch):
				Class object' config.
		
		Returns:
			instance (object, optional):
				An instance of the class that is created.
		"""
		if cfg is None:
			return None
		
		if not isinstance(cfg, (dict, Munch)):
			error_console.log("`cfg` must be a dict.")
			return None
		
		if "name" not in cfg:
			error_console.log("`cfg` dict must contain the key `name`.")
			return None
		
		cfg_    = deepcopy(cfg)
		name    = cfg_.pop("name")
		cfg_   |= kwargs
		return self.build(name=name, **cfg_)
	
	def build_from_dictlist(
		self, cfgs: Optional[list[Union[dict, Munch]]], **kwargs
	) -> Optional[list[object]]:
		"""Factory command to create instances of classes. This method gets the
		appropriate classes from the registry while passing in the parameters
		given in `cfgs`.

		Args:
			cfgs (list[dict, Munch], optional):
				List of class objects' configs.

		Returns:
			instances (list[object], optional):
				Instances of the classes that are created.
		"""
		if cfgs is None:
			return None
		
		if (not is_list_of(cfgs, expected_type=dict)
			and not is_list_of(cfgs, expected_type=Munch)):
			error_console.log("`cfgs` must be a list of dict.")
			return None
		
		cfgs_     = deepcopy(cfgs)
		instances = []
		for cfg in cfgs_:
			name  = cfg.pop("name")
			cfg  |= kwargs
			instances.append(self.build(name=name, **cfg))
		
		return instances if len(instances) > 0 else None
