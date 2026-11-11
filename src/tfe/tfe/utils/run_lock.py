#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Allow only one pipeline run at a time on a data root.

Every run rewrites ``<data_dir>/<dataset>/temps/<camera>.txt`` (opened with "w",
tfe/utils/aic_result_writer.py:117-118), so two concurrent runs would corrupt
each other's results. The web GUI jobs and the GPU parity test hold this lock.
This module must not import torch.
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from typing import Iterator

__all__ = [
	"run_lock"
]


# MARK: - Lock

@contextmanager
def run_lock(root_dir: str) -> Iterator[None]:
	"""Hold a non-blocking exclusive ``flock`` on the directory ``<root_dir>/data``.

	The lock is taken on the directory itself, so no lock file is created, and
	the kernel releases it when the descriptor closes, even if the process dies.

	Args:
		root_dir (str):
			The root that contains ``data/`` (``src/tfe`` by default).

	Raises:
		BlockingIOError: another holder, in this or another process, has the lock.
		FileNotFoundError: ``<root_dir>/data`` does not exist.
	"""
	fd = os.open(os.path.join(root_dir, "data"), os.O_RDONLY)
	try:
		fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
		try:
			yield
		finally:
			fcntl.flock(fd, fcntl.LOCK_UN)
	finally:
		os.close(fd)
