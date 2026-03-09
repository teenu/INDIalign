#!/usr/bin/env python3
"""Compatibility shim for the modular GPU local validation implementation."""

from __future__ import annotations

import sys

from scoring._local_validation_gpu import runtime as _runtime

sys.modules[__name__] = _runtime

if __name__ == "__main__":
    _runtime.main()
