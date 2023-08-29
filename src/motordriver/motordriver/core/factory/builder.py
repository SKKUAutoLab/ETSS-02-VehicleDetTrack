#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory classes.
"""

from __future__ import annotations

from core.factory.factory import Factory

__all__ = [
	"CAMERAS",
	"DETECTORS",
	"IDENTIFICATIONS",
	"MATCHERS",

	"FILE_HANDLERS",

	"AUGMENTS",
	"TRANSFORMS"
]

# MARK: - Modules

CAMERAS         = Factory(name="cameras")
DETECTORS       = Factory(name="object_detectors")
IDENTIFICATIONS = Factory(name="identifications")
TRACKERS        = Factory(name="trackers")
MATCHERS        = Factory(name="matchers")

# MARK: - File

FILE_HANDLERS   = Factory(name="file_handler")

# MARK: - Augment

AUGMENTS        = Factory(name="augments")
TRANSFORMS      = Factory(name="transforms")
