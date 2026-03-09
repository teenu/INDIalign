#!/usr/bin/env python3
"""Compatibility shim for the modular GPU TM-score implementation."""

from __future__ import annotations

from scoring._tmscore_gpu import runtime as _runtime

import sys

sys.modules[__name__] = _runtime
