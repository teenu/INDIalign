#!/usr/bin/env python3
"""Runtime state and public API for GPU local TM-score validation."""

from __future__ import annotations

import torch

import scoring.tmscore_gpu as tg

INVALID_COORD = -1e6
MAX_NATIVE_FRAMES = 40
MAX_PRED_FRAMES = 5
DEFAULT_MAX_ITER = 20
DEFAULT_USE_FRAGMENT_SEARCH = True
DEFAULT_MAX_MEM_GB = 20.0
DEFAULT_DP_ITER = 0
DEFAULT_BACKEND_MODE = "strict"
_DEFAULT_DTYPE = tg._DEFAULT_DTYPE

d0_from_length = tg.d0_from_length
tmscore_search = tg.tmscore_search
kabsch_batch = tg.kabsch_batch
_apply_transform = tg._apply_transform
_tm_score = tg._tm_score

from . import backend as _backend
from . import coords as _coords
from . import multimer as _multimer
from . import scoring as _scoring
from . import cli as _cli

_configure_tmscore_backend = _backend._configure_tmscore_backend
_effective_triton_paths = _backend._effective_triton_paths

_sorted_by_resid = _coords._sorted_by_resid
_available_frames = _coords._available_frames
extract_coords = _coords.extract_coords

_score_pair_batches = _scoring._score_pair_batches
_multimer_group_indices = _multimer._multimer_group_indices
_group_targets_by_length = _multimer._group_targets_by_length
_icp_chain_candidates = _multimer._icp_chain_candidates
_hungarian_chain_score = _multimer._hungarian_chain_score

score_target = _scoring.score_target
score_parallel = _scoring.score_parallel
main = _cli.main
