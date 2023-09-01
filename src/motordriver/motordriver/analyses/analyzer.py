from __future__ import annotations

import abc
from typing import Optional

__all__ = [
	"BaseAnalyzer"
]

# MARK: - BaseAnalyzer


# noinspection PyShadowingBuiltins
class BaseAnalyzer(metaclass=abc.ABCMeta):
	"""Base Analyzer.

	Attributes:
		name (str):
			Name of the analyzer model.
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			name           : Optional[str] = None,
			*args, **kwargs
	):
		super().__init__()
		self.name = name

	def update(self):
		pass
