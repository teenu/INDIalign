"""Command-line interface for GPU local validation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from . import runtime as rt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU local TM-score validation."
    )
    parser.add_argument("submission", type=str, help="Path to submission.csv")
    parser.add_argument("--validation", type=str, default=None, help="Path to validation_labels.csv")
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: cuda if available)")
    fs_group = parser.add_mutually_exclusive_group()
    fs_group.add_argument(
        "--fragment-search",
        action="store_true",
        help="Enable fragment-seed search (default).",
    )
    fs_group.add_argument(
        "--no-fragment-search",
        action="store_true",
        help="Disable fragment-seed search for faster but less accurate scoring.",
    )
    parser.add_argument("--max-iter", type=int, default=None, help="Refinement iterations per seed (default: 20, 8 for ultrafast)")
    parser.add_argument("--max-mem-gb", type=float, default=rt.DEFAULT_MAX_MEM_GB, help="GPU memory budget in GB")
    parser.add_argument("--dp-iter", type=int, default=rt.DEFAULT_DP_ITER, help="NW DP alignment iterations (0=off)")
    parser.add_argument("--float64", action="store_true", help="Use float64 precision (slower, for validation)")
    parser.add_argument(
        "--backend-mode",
        choices=("strict", "hybrid", "fast", "ultrafast"),
        default=rt.DEFAULT_BACKEND_MODE,
        help=(
            "Validation backend preset: strict=parity-safe torch path (default), "
            "hybrid=Torch Kabsch + Triton refine parity guard, fast=Triton Kabsch+refine, "
            "ultrafast=fast + reduced seeds/iters/fracs (max_iter=8, 2 mults, 1 frac)."
        ),
    )
    parser.add_argument(
        "--triton-refine",
        action="store_true",
        help="Override backend mode to enable Triton fused selection kernel in iterative seed refine.",
    )
    parser.add_argument(
        "--triton-kabsch",
        action="store_true",
        help="Override backend mode to enable Triton fused Kabsch kernel (fastest, may change float32 scores).",
    )
    parser.add_argument(
        "--triton-exact-parity",
        action="store_true",
        help="Enable Triton refine with exact-score parity guard (keeps Triton Kabsch off; tuned for parity on validation).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    validation_path = (
        Path(args.validation)
        if args.validation
        else project_root.parent / "stanford-rna-3d-folding-2" / "validation_labels.csv"
    )

    submission = pd.read_csv(args.submission)
    validation = pd.read_csv(validation_path)

    targets = submission["ID"].str.rsplit("_", n=1).str[0].nunique()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targets: {targets}")
    print(f"Device: {dev}")

    if args.float64:
        import scoring.tmscore_gpu as _tg
        _tg._DEFAULT_DTYPE = torch.float64
        rt._DEFAULT_DTYPE = torch.float64

    if args.triton_exact_parity and args.triton_kabsch:
        print("Triton exact parity mode disables Triton Kabsch to preserve score parity.")
    backend_summary = rt._configure_tmscore_backend(
        args.backend_mode,
        triton_refine=bool(args.triton_refine),
        triton_kabsch=bool(args.triton_kabsch),
        triton_exact_parity=bool(args.triton_exact_parity),
    )
    print(backend_summary)
    dtype_in_use = rt._DEFAULT_DTYPE
    eff = rt._effective_triton_paths(dev, dtype_in_use)
    if args.float64 and args.backend_mode != "strict":
        print(
            "Note: --float64 disables Triton kernels; non-strict modes keep their "
            "search heuristics but execute on torch kernels."
        )
    print(
        "Effective Triton paths | "
        f"has_triton={eff['has_triton']} cuda_device={eff['cuda_device']} "
        f"dtype={dtype_in_use} dtype_ok={eff['dtype_ok']} | "
        f"refine={eff['refine_effective']} kabsch={eff['kabsch_effective']} "
        f"score={eff['score_effective']}"
    )

    use_fragment_search = rt.DEFAULT_USE_FRAGMENT_SEARCH
    if args.fragment_search:
        use_fragment_search = True
    if args.no_fragment_search:
        use_fragment_search = False

    max_iter = args.max_iter

    if dev.startswith("cuda"):
        torch.cuda.synchronize(device=dev)
    t0 = time.time()
    mean_tm, per_target = rt.score_parallel(
        validation,
        submission,
        device=dev,
        max_iter=max_iter,
        use_fragment_search=use_fragment_search,
        max_mem_gb=float(args.max_mem_gb),
        dp_iter=int(args.dp_iter),
    )
    if dev.startswith("cuda"):
        torch.cuda.synchronize(device=dev)
    dt = time.time() - t0
    for tid in sorted(per_target):
        print(f"TM:{tid}={per_target[tid]:.6f}")
    print(f"Mean TM-score: {mean_tm:.6f}")
    print(f"Elapsed: {dt:.2f}s")
