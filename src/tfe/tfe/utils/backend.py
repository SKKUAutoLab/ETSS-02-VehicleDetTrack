#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from enum import Enum


__all__ = [
	"VISION_BACKEND",
	"VisionBackend",
	"interpolation_vision_backend_from_int",
	"interpolation_vision_backend_from_str",
]


# MARK: - VisionBackend

class VisionBackend(Enum):
	CV      = "cv"
	LIBVIPS = "libvips"
	PIL     = "pil"


VISION_BACKEND = VisionBackend.PIL


def interpolation_vision_backend_from_int(i: int) -> VisionBackend:
	inverse_backend_mapping = {
		0: VisionBackend.CV,
		1: VisionBackend.LIBVIPS,
		2: VisionBackend.PIL,
	}
	return inverse_backend_mapping[i]


def interpolation_vision_backend_from_str(i: str) -> VisionBackend:
	inverse_backend_mapping = {
		"cv"     : VisionBackend.CV,
		"libvips": VisionBackend.LIBVIPS,
		"pil"    : VisionBackend.PIL,
	}
	return inverse_backend_mapping[i]
