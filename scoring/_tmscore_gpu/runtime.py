#!/usr/bin/env python3
"""Runtime state and public API for the GPU TM-score engine."""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

_DEFAULT_DTYPE = torch.float32
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

_D0_SEARCH_MULTS = (2.0, 1.5, 1.0, 0.75, 0.5, 0.25)
_D0_SEARCH_ADD_OFFSETS: tuple[float, ...] = ()
_CONTACT_SEED_MULTS: tuple[float, ...] = ()
_CONTACT_SEED_ADD_OFFSETS: tuple[float, ...] = ()
_CONTACT_SEED_TOL_MULT: float = 0.25
_CONTACT_SEED_MAX_ANCHORS: int = 0
_TM_WEIGHTED_REFINE_ITERS: int = 0
_TM_WEIGHTED_REFINE_TOPK: int = 0
_TM_WEIGHTED_REFINE_SCORE_MARGIN: float = 0.0
_PAIRSEED_KABSCH_BATCH_CAP: int = 8192
_MULTS_CACHE: dict[tuple[str, int, torch.dtype], torch.Tensor] = {}
_PAIRSEED_BYTES_PER_RES_FLOAT = 96
_PAIRSEED_BYTES_PER_RES_BOOL = 8
_EIGH_CHUNK = 30000
_ENABLE_TRITON_KABSCH = os.environ.get("BAMBOO_TRITON_KABSCH", "0") != "0"
_ENABLE_TRITON_REFINE = os.environ.get("BAMBOO_TRITON_REFINE", "0") != "0"
_TRITON_MIN_B = 32
_TRITON_MIN_N = 32
_TRITON_REFINE_PARITY_GUARD = os.environ.get("BAMBOO_TRITON_REFINE_PARITY_GUARD", "1") != "0"
_TRITON_REFINE_PARITY_ABS_TOL = float(os.environ.get("BAMBOO_TRITON_REFINE_PARITY_ABS_TOL", "5e-3"))
_TRITON_REFINE_PARITY_REL_TOL = float(os.environ.get("BAMBOO_TRITON_REFINE_PARITY_REL_TOL", "1e-5"))
_DIST_FRACS: tuple[float, ...] = (0.25, 0.5, 0.75)
_MAX_FRAG_STARTS: int | None = None
_MAX_ITER: int = 20
_SEARCH_SELECTION_USES_SCORE_D8 = True
_ENABLE_TRITON_SCORE = os.environ.get("BAMBOO_TRITON_SCORE", "0") != "0"
_CUDA_CAP_CACHE: dict[int, tuple[int, int]] = {}

_DEFAULT_ENABLE_TRITON_KABSCH = _ENABLE_TRITON_KABSCH
_DEFAULT_ENABLE_TRITON_REFINE = _ENABLE_TRITON_REFINE
_DEFAULT_ENABLE_TRITON_SCORE = _ENABLE_TRITON_SCORE
_DEFAULT_TRITON_REFINE_PARITY_GUARD = _TRITON_REFINE_PARITY_GUARD
_DEFAULT_TRITON_REFINE_PARITY_ABS_TOL = _TRITON_REFINE_PARITY_ABS_TOL
_DEFAULT_TRITON_REFINE_PARITY_REL_TOL = _TRITON_REFINE_PARITY_REL_TOL
_DEFAULT_D0_SEARCH_MULTS = tuple(_D0_SEARCH_MULTS)
_DEFAULT_D0_SEARCH_ADD_OFFSETS = tuple(_D0_SEARCH_ADD_OFFSETS)
_DEFAULT_CONTACT_SEED_MULTS = tuple(_CONTACT_SEED_MULTS)
_DEFAULT_CONTACT_SEED_ADD_OFFSETS = tuple(_CONTACT_SEED_ADD_OFFSETS)
_DEFAULT_CONTACT_SEED_TOL_MULT = _CONTACT_SEED_TOL_MULT
_DEFAULT_CONTACT_SEED_MAX_ANCHORS = _CONTACT_SEED_MAX_ANCHORS
_DEFAULT_DIST_FRACS = tuple(_DIST_FRACS)
_DEFAULT_MAX_FRAG_STARTS = _MAX_FRAG_STARTS
_DEFAULT_MAX_ITER = _MAX_ITER
_DEFAULT_SEARCH_SELECTION_USES_SCORE_D8 = _SEARCH_SELECTION_USES_SCORE_D8
_DEFAULT_TM_WEIGHTED_REFINE_ITERS = _TM_WEIGHTED_REFINE_ITERS
_DEFAULT_TM_WEIGHTED_REFINE_TOPK = _TM_WEIGHTED_REFINE_TOPK
_DEFAULT_TM_WEIGHTED_REFINE_SCORE_MARGIN = _TM_WEIGHTED_REFINE_SCORE_MARGIN


def reset_backend_runtime_state() -> None:
    """Reset mutable backend globals to canonical import-time defaults."""
    global _ENABLE_TRITON_KABSCH, _ENABLE_TRITON_REFINE, _ENABLE_TRITON_SCORE
    global _TRITON_REFINE_PARITY_GUARD, _TRITON_REFINE_PARITY_ABS_TOL, _TRITON_REFINE_PARITY_REL_TOL
    global _D0_SEARCH_MULTS, _D0_SEARCH_ADD_OFFSETS, _CONTACT_SEED_MULTS, _CONTACT_SEED_ADD_OFFSETS
    global _CONTACT_SEED_TOL_MULT, _CONTACT_SEED_MAX_ANCHORS
    global _DIST_FRACS, _MAX_FRAG_STARTS, _MAX_ITER, _SEARCH_SELECTION_USES_SCORE_D8
    global _TM_WEIGHTED_REFINE_ITERS, _TM_WEIGHTED_REFINE_TOPK, _TM_WEIGHTED_REFINE_SCORE_MARGIN

    _ENABLE_TRITON_KABSCH = _DEFAULT_ENABLE_TRITON_KABSCH
    _ENABLE_TRITON_REFINE = _DEFAULT_ENABLE_TRITON_REFINE
    _ENABLE_TRITON_SCORE = _DEFAULT_ENABLE_TRITON_SCORE
    _TRITON_REFINE_PARITY_GUARD = _DEFAULT_TRITON_REFINE_PARITY_GUARD
    _TRITON_REFINE_PARITY_ABS_TOL = _DEFAULT_TRITON_REFINE_PARITY_ABS_TOL
    _TRITON_REFINE_PARITY_REL_TOL = _DEFAULT_TRITON_REFINE_PARITY_REL_TOL
    _D0_SEARCH_MULTS = tuple(_DEFAULT_D0_SEARCH_MULTS)
    _D0_SEARCH_ADD_OFFSETS = tuple(_DEFAULT_D0_SEARCH_ADD_OFFSETS)
    _CONTACT_SEED_MULTS = tuple(_DEFAULT_CONTACT_SEED_MULTS)
    _CONTACT_SEED_ADD_OFFSETS = tuple(_DEFAULT_CONTACT_SEED_ADD_OFFSETS)
    _CONTACT_SEED_TOL_MULT = _DEFAULT_CONTACT_SEED_TOL_MULT
    _CONTACT_SEED_MAX_ANCHORS = _DEFAULT_CONTACT_SEED_MAX_ANCHORS
    _DIST_FRACS = tuple(_DEFAULT_DIST_FRACS)
    _MAX_FRAG_STARTS = _DEFAULT_MAX_FRAG_STARTS
    _MAX_ITER = _DEFAULT_MAX_ITER
    _SEARCH_SELECTION_USES_SCORE_D8 = _DEFAULT_SEARCH_SELECTION_USES_SCORE_D8
    _TM_WEIGHTED_REFINE_ITERS = _DEFAULT_TM_WEIGHTED_REFINE_ITERS
    _TM_WEIGHTED_REFINE_TOPK = _DEFAULT_TM_WEIGHTED_REFINE_TOPK
    _TM_WEIGHTED_REFINE_SCORE_MARGIN = _DEFAULT_TM_WEIGHTED_REFINE_SCORE_MARGIN
    _MULTS_CACHE.clear()


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _cuda_capability(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    idx = 0 if device.index is None else int(device.index)
    out = _CUDA_CAP_CACHE.get(idx)
    if out is None:
        out = torch.cuda.get_device_capability(idx)
        _CUDA_CAP_CACHE[idx] = out
    return out


def _is_blackwell(device: torch.device) -> bool:
    cap = _cuda_capability(device)
    return bool(cap is not None and cap[0] >= 12)


def d0_from_length(
    L: torch.Tensor | int | float,
    mol: str = "rna",
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (d0_final, d0_search, score_d8) for lengths L.

    Parameters
    ----------
    L:
        Scalar or tensor of effective alignment lengths.
    mol:
        Molecule type. Only "rna" is currently supported.
    dtype:
        Output dtype. Defaults to ``_DEFAULT_DTYPE`` (float32).
    """
    if str(mol).lower() != "rna":
        raise ValueError("Only mol='rna' is supported.")

    if dtype is None:
        dtype = _DEFAULT_DTYPE

    if not torch.is_tensor(L):
        Lt = torch.as_tensor(L, dtype=dtype)
    else:
        Lt = L.to(dtype=dtype)

    Lc = torch.clamp(Lt, min=1.0)

    # Bucket index: 0→≤11, 1→(11,15], 2→(15,19], 3→(19,23], 4→>23
    bucket = (Lc > 11.0).long() + (Lc > 15.0).long() + (Lc > 19.0).long() + (Lc > 23.0).long()
    lut = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7], dtype=dtype, device=Lc.device)
    d0 = lut[bucket]
    big = Lc >= 30.0
    d0_big = 0.6 * torch.sqrt(torch.clamp(Lc - 0.5, min=1e-12)) - 2.5
    d0 = torch.where(big, d0_big, d0)
    d0 = torch.clamp(d0, min=0.3)

    # USalign-style search heuristics.
    d0_search = 1.24 * torch.pow(torch.clamp(Lc - 15.0, min=0.0), 1.0 / 3.0) - 1.8
    d0_search = torch.clamp(d0_search, min=4.5, max=8.0)
    score_d8 = 1.5 * torch.pow(Lc, 0.3) + 3.5

    return d0, d0_search, score_d8


def _device_cache_key(device: torch.device, dtype: torch.dtype) -> tuple[str, int, torch.dtype]:
    return (device.type, -1 if device.index is None else int(device.index), dtype)


def _d0_mults_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = _device_cache_key(device, dtype)
    out = _MULTS_CACHE.get(key)
    if out is None or out.shape[0] != len(_D0_SEARCH_MULTS):
        out = torch.tensor(_D0_SEARCH_MULTS, dtype=dtype, device=device)
        _MULTS_CACHE[key] = out
    return out


def configure_pair_local_search(
    *,
    d0_add_offsets: tuple[float, ...] | None = None,
    contact_seed_mults: tuple[float, ...] | None = None,
    contact_seed_add_offsets: tuple[float, ...] | None = None,
    contact_seed_tol_mult: float | None = None,
    contact_seed_max_anchors: int | None = None,
    search_uses_score_d8: bool | None = None,
    tm_weighted_refine_iters: int | None = None,
    tm_weighted_refine_topk: int | None = None,
    tm_weighted_refine_score_margin: float | None = None,
) -> None:
    """Configure optional pair-local search expansions for pure GPU scoring."""
    global _D0_SEARCH_ADD_OFFSETS, _CONTACT_SEED_MULTS, _CONTACT_SEED_ADD_OFFSETS
    global _CONTACT_SEED_TOL_MULT, _CONTACT_SEED_MAX_ANCHORS
    global _SEARCH_SELECTION_USES_SCORE_D8, _TM_WEIGHTED_REFINE_ITERS
    global _TM_WEIGHTED_REFINE_TOPK, _TM_WEIGHTED_REFINE_SCORE_MARGIN

    if d0_add_offsets is not None:
        _D0_SEARCH_ADD_OFFSETS = tuple(float(v) for v in d0_add_offsets if float(v) >= 0.0)
    if contact_seed_mults is not None:
        _CONTACT_SEED_MULTS = tuple(float(v) for v in contact_seed_mults if float(v) > 0.0)
    if contact_seed_add_offsets is not None:
        _CONTACT_SEED_ADD_OFFSETS = tuple(float(v) for v in contact_seed_add_offsets if float(v) >= 0.0)
    if contact_seed_tol_mult is not None:
        _CONTACT_SEED_TOL_MULT = max(0.0, float(contact_seed_tol_mult))
    if contact_seed_max_anchors is not None:
        _CONTACT_SEED_MAX_ANCHORS = max(0, int(contact_seed_max_anchors))
    if search_uses_score_d8 is not None:
        _SEARCH_SELECTION_USES_SCORE_D8 = bool(search_uses_score_d8)
    if tm_weighted_refine_iters is not None:
        _TM_WEIGHTED_REFINE_ITERS = max(0, int(tm_weighted_refine_iters))
    if tm_weighted_refine_topk is not None:
        _TM_WEIGHTED_REFINE_TOPK = max(0, int(tm_weighted_refine_topk))
    if tm_weighted_refine_score_margin is not None:
        _TM_WEIGHTED_REFINE_SCORE_MARGIN = max(0.0, float(tm_weighted_refine_score_margin))
    if 1.0 in _CONTACT_SEED_MULTS and 0.0 in _CONTACT_SEED_ADD_OFFSETS:
        _CONTACT_SEED_ADD_OFFSETS = tuple(v for v in _CONTACT_SEED_ADD_OFFSETS if v != 0.0)


def _d0_search_candidates(d0_search: torch.Tensor) -> torch.Tensor:
    """Return pair-local search radii from multiplicative and additive families."""
    device = d0_search.device
    dtype = d0_search.dtype
    families = [d0_search.unsqueeze(1) * _d0_mults_tensor(device, dtype).unsqueeze(0)]
    if _D0_SEARCH_ADD_OFFSETS:
        add = torch.tensor(_D0_SEARCH_ADD_OFFSETS, dtype=dtype, device=device)
        families.append(d0_search.unsqueeze(1) + add.unsqueeze(0))
    if len(families) == 1:
        return families[0]
    return torch.cat(families, dim=1)


def _contact_seed_radii(d0_search: torch.Tensor) -> torch.Tensor:
    """Return pair-local radii for local-contact seed families."""
    device = d0_search.device
    dtype = d0_search.dtype
    families: list[torch.Tensor] = []
    if _CONTACT_SEED_MULTS:
        mults = torch.tensor(_CONTACT_SEED_MULTS, dtype=dtype, device=device)
        families.append(d0_search.unsqueeze(1) * mults.unsqueeze(0))
    if _CONTACT_SEED_ADD_OFFSETS:
        add = torch.tensor(_CONTACT_SEED_ADD_OFFSETS, dtype=dtype, device=device)
        families.append(d0_search.unsqueeze(1) + add.unsqueeze(0))
    if not families:
        return torch.zeros((d0_search.shape[0], 0), dtype=dtype, device=device)
    if len(families) == 1:
        return families[0]
    return torch.cat(families, dim=1)


def _pairseed_chunk_plan(
    B: int,
    N: int,
    KM: int,
    dtype: torch.dtype,
    max_mem_gb: float,
) -> tuple[int, int]:
    """Return (chunk_b, chunk_km) for expanded pair-seed batches.

    The estimate is intentionally conservative and reserves headroom because
    `_iterative_seed_refine` creates several transient tensors per iteration.
    """
    if B <= 0 or KM <= 0:
        return 1, 1

    itemsize = torch.empty((), dtype=dtype).element_size()
    per_pairseed = max(
        1,
        N * (_PAIRSEED_BYTES_PER_RES_FLOAT + _PAIRSEED_BYTES_PER_RES_BOOL)
        + itemsize * 64,
    )
    budget = max(1, int(max_mem_gb * (1024 ** 3) * 0.55))
    max_pairseeds = max(1, budget // per_pairseed)
    max_pairseeds = min(max_pairseeds, _PAIRSEED_KABSCH_BATCH_CAP)

    if B * KM <= max_pairseeds:
        return B, KM

    if N <= 512:
        km_cap = 1024
    elif N <= 2048:
        km_cap = 512
    else:
        km_cap = 256
    km_cap = min(km_cap, KM)

    chunk_b = min(B, max(1, max_pairseeds // max(1, km_cap)))
    chunk_km = min(KM, max(1, max_pairseeds // max(1, chunk_b)))
    chunk_km = min(chunk_km, km_cap)
    chunk_b = min(B, max(1, max_pairseeds // max(1, chunk_km)))
    return chunk_b, chunk_km


# ---------------------------------------------------------------------------
# Re-exports from submodules.  Imported here so that downstream code can
# access everything through the runtime module (which replaces the original
# monolithic scoring.tmscore_gpu namespace via sys.modules).
# ---------------------------------------------------------------------------

from . import triton_ops as _triton_ops
from . import rigid as _rigid
from . import dp as _dp
from . import search as _search

_triton_select_launch_config = _triton_ops._triton_select_launch_config
_triton_kabsch_launch_config = _triton_ops._triton_kabsch_launch_config
_can_use_triton_kabsch = _triton_ops._can_use_triton_kabsch
_can_use_triton_refine = _triton_ops._can_use_triton_refine
_kabsch_fused_triton = _triton_ops._kabsch_fused_triton
_select_mask_triton = _triton_ops._select_mask_triton
_can_use_triton_score = _triton_ops._can_use_triton_score
_triton_score_launch_config = _triton_ops._triton_score_launch_config
_tm_score_fused_triton = _triton_ops._tm_score_fused_triton
_dist2_fused_triton = _triton_ops._dist2_fused_triton

_eigh_largest_eigvec_4x4 = _rigid._eigh_largest_eigvec_4x4
_kabsch_finalize_from_cov = _rigid._kabsch_finalize_from_cov
_kabsch_batch_torch = _rigid._kabsch_batch_torch
_kabsch_batch_weighted = _rigid._kabsch_batch_weighted
kabsch_batch = _rigid.kabsch_batch
_apply_transform = _rigid._apply_transform
_tm_score_impl = _rigid._tm_score_impl
_tm_score = _tm_score_impl
_tm_score_fused = _rigid._tm_score_fused
_weighted_refine_chunk_size = _rigid._weighted_refine_chunk_size
_dist2_fused = _rigid._dist2_fused
_tm_weighted_refine = _rigid._tm_weighted_refine
_weighted_refine_candidate_mask = _rigid._weighted_refine_candidate_mask
_tm_weighted_refine_refines_all = _rigid._tm_weighted_refine_refines_all
_tm_weighted_refine_is_selective = _rigid._tm_weighted_refine_is_selective
_merge_best_transforms = _rigid._merge_best_transforms

_fragment_lengths = _search._fragment_lengths
_start_positions = _search._start_positions
_build_seed_windows = _search._build_seed_windows
_build_pairwise_seed_masks = _search._build_pairwise_seed_masks
_build_pairwise_topk_seed_masks = _search._build_pairwise_topk_seed_masks
_build_pairwise_contact_seed_masks = _search._build_pairwise_contact_seed_masks
_build_pairwise_anchor_contact_seed_masks = _search._build_pairwise_anchor_contact_seed_masks
_evaluate_seed_bank = _search._evaluate_seed_bank
_iterative_seed_refine = _search._iterative_seed_refine

_score_matrix = _dp._score_matrix
_nw_dp = _dp._nw_dp
_nw_traceback = _dp._nw_traceback
_dp_refine_chunk = _dp._dp_refine_chunk
_dp_refine = _dp._dp_refine

tmscore_search = _search.tmscore_search
