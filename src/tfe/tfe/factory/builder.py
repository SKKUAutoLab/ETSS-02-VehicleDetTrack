#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory classes.
"""

from __future__ import annotations

from thermal_pedestrian.core.factory.factory import Factory

__all__ = [
	"CAMERAS",
	"DETECTORS",
	"TRACKERS",
	"MATCHERS",
	"ANALYZERS",
	"HEURISTICS",

	"FILE_HANDLERS",

	"AUGMENTS",
	"TRANSFORMS"
]

# MARK: - Modules

CAMERAS         = Factory(name="cameras")
DETECTORS       = Factory(name="object_detectors")
IDENTIFICATIONS = Factory(name="identifications")
HEURISTICS      = Factory(name="heuristics")
TRACKERS        = Factory(name="trackers")
MATCHERS        = Factory(name="matchers")
ANALYZERS       = Factory(name="analyzers")

# MARK: - File

FILE_HANDLERS   = Factory(name="file_handler")

# MARK: - Augment

AUGMENTS        = Factory(name="augments")
TRANSFORMS      = Factory(name="transforms")
