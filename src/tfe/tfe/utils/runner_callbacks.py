#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Callbacks through which the runners report to their host, the web GUI's session.

This module must not import torch: the web layer imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = [
	"RunnerCallbacks"
]


def _ignore(*args, **kwargs) -> None:
	"""Default callback: do nothing."""


# MARK: - RunnerCallbacks

@dataclass
class RunnerCallbacks:
	"""Hooks the runner calls from its own thread.

	Attributes:
		on_info (Callable[[dict], None]):
			Called with the dict ``CameraQT6`` emits: one per drawn frame
			(``is_run`` True), then a final one (``is_run`` False) after the
			result file is closed.
		on_finished (Callable[[bool], None]):
			Called once after a run that raised nothing; the argument is True
			when the stop event ended the run early.
		on_error (Callable[[BaseException], None]):
			Called with the exception before ``run()`` re-raises it.
	"""
	on_info    : Callable[[dict], None]          = _ignore
	on_finished: Callable[[bool], None]          = _ignore
	on_error   : Callable[[BaseException], None] = _ignore
