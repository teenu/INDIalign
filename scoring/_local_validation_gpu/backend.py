"""Backend configuration helpers for GPU local validation."""

from __future__ import annotations

import torch


def _configure_tmscore_backend(
    mode: str,
    *,
    triton_refine: bool = False,
    triton_kabsch: bool = False,
    triton_exact_parity: bool = False,
) -> str:
    """Apply validation backend preset + overrides to scoring.tmscore_gpu."""
    import scoring.tmscore_gpu as _tg

    mode = str(mode).strip().lower()
    if mode not in {"strict", "hybrid", "fast", "ultrafast"}:
        raise ValueError(f"Unknown backend mode: {mode!r}")

    _MODE_PRESETS = {
        "strict": {
            "_ENABLE_TRITON_REFINE": False, "_ENABLE_TRITON_KABSCH": False,
            "_ENABLE_TRITON_SCORE": False,
        },
        "hybrid": {
            "_ENABLE_TRITON_REFINE": True, "_ENABLE_TRITON_KABSCH": False,
            "_ENABLE_TRITON_SCORE": True, "_TRITON_REFINE_PARITY_GUARD": True,
            "_TRITON_REFINE_PARITY_ABS_TOL": 0.45,
        },
        "fast": {
            "_ENABLE_TRITON_REFINE": True, "_ENABLE_TRITON_KABSCH": True,
            "_ENABLE_TRITON_SCORE": True, "_TRITON_REFINE_PARITY_GUARD": False,
        },
        "ultrafast": {
            "_ENABLE_TRITON_REFINE": True, "_ENABLE_TRITON_KABSCH": True,
            "_ENABLE_TRITON_SCORE": True, "_TRITON_REFINE_PARITY_GUARD": False,
            "_D0_SEARCH_MULTS": (1.0, 0.5), "_DIST_FRACS": (0.5,),
            "_MAX_FRAG_STARTS": 20, "_MAX_ITER": 8,
        },
    }

    # Reset all mutable backend globals so mode switching is process-safe.
    if hasattr(_tg, "reset_backend_runtime_state"):
        _tg.reset_backend_runtime_state()

    for k, v in _MODE_PRESETS[mode].items():
        setattr(_tg, k, v)
    if mode == "ultrafast":
        _tg._MULTS_CACHE.clear()

    if triton_refine:
        _tg._ENABLE_TRITON_REFINE = True
    if triton_kabsch:
        _tg._ENABLE_TRITON_KABSCH = True

    if triton_exact_parity:
        _tg._ENABLE_TRITON_REFINE = True
        _tg._ENABLE_TRITON_KABSCH = False
        if hasattr(_tg, "_TRITON_REFINE_PARITY_GUARD"):
            _tg._TRITON_REFINE_PARITY_GUARD = True
        if hasattr(_tg, "_TRITON_REFINE_PARITY_ABS_TOL"):
            _tg._TRITON_REFINE_PARITY_ABS_TOL = 0.45

    return (
        f"Backend mode: {mode} | "
        f"triton_refine={getattr(_tg, '_ENABLE_TRITON_REFINE', False)} "
        f"triton_kabsch={getattr(_tg, '_ENABLE_TRITON_KABSCH', False)} "
        f"triton_score={getattr(_tg, '_ENABLE_TRITON_SCORE', False)} | "
        f"parity_guard={getattr(_tg, '_TRITON_REFINE_PARITY_GUARD', False)} "
        f"min_B={getattr(_tg, '_TRITON_MIN_B', 'n/a')} "
        f"min_N={getattr(_tg, '_TRITON_MIN_N', 'n/a')} | "
        f"d0_mults={getattr(_tg, '_D0_SEARCH_MULTS', ())} "
        f"dist_fracs={getattr(_tg, '_DIST_FRACS', ())} "
        f"max_frag_starts={getattr(_tg, '_MAX_FRAG_STARTS', None)}"
    )


def _effective_triton_paths(device: str, dtype: torch.dtype) -> dict[str, bool]:
    """Return configured vs effectively-active Triton paths for this run."""
    import scoring.tmscore_gpu as _tg

    has_triton = bool(getattr(_tg, "_HAS_TRITON", False))
    cuda_device = str(device).startswith("cuda")
    dtype_ok = dtype == torch.float32
    runtime_ok = has_triton and cuda_device and dtype_ok
    refine_cfg = bool(getattr(_tg, "_ENABLE_TRITON_REFINE", False))
    kabsch_cfg = bool(getattr(_tg, "_ENABLE_TRITON_KABSCH", False))
    score_cfg = bool(getattr(_tg, "_ENABLE_TRITON_SCORE", False))
    return {
        "has_triton": has_triton,
        "cuda_device": cuda_device,
        "dtype_ok": dtype_ok,
        "refine_configured": refine_cfg,
        "kabsch_configured": kabsch_cfg,
        "score_configured": score_cfg,
        "refine_effective": refine_cfg and runtime_ok,
        "kabsch_effective": kabsch_cfg and runtime_ok,
        "score_effective": score_cfg and runtime_ok,
    }
