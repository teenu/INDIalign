#!/usr/bin/env python3
"""Head-to-head comparison: GPU TM-score vs USalign.

Runs both scorers on the same submission and produces per-target analysis
to determine whether GPU-parallel seed search systematically finds better
superpositions than USalign's serial heuristics.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def run_usalign_scoring(
    validation: pd.DataFrame,
    submission: pd.DataFrame,
    usalign_bin: str,
    workers: int,
) -> tuple[float, dict[str, float], float]:
    from scoring.local_validation_mt import score_parallel
    t0 = time.time()
    mean_tm, per_target = score_parallel(
        validation, submission, usalign_bin, workers, mode="thread",
    )
    dt = time.time() - t0
    return mean_tm, per_target, dt


def run_gpu_scoring(
    validation: pd.DataFrame,
    submission: pd.DataFrame,
    device: str,
    backend_mode: str,
    use_float64: bool,
    dp_iter: int = 0,
) -> tuple[float, dict[str, float], float]:
    import scoring.tmscore_gpu as _tg
    import scoring.local_validation_gpu as _lvg
    from scoring.local_validation_gpu import (
        _configure_tmscore_backend,
        score_parallel,
    )

    if use_float64:
        _tg._DEFAULT_DTYPE = torch.float64
        _lvg._DEFAULT_DTYPE = torch.float64
    else:
        _tg._DEFAULT_DTYPE = torch.float32
        _lvg._DEFAULT_DTYPE = torch.float32

    _configure_tmscore_backend(backend_mode)

    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)
    t0 = time.time()
    mean_tm, per_target = score_parallel(
        validation, submission,
        device=device,
        max_iter=None,
        use_fragment_search=True,
        max_mem_gb=20.0,
        dp_iter=dp_iter,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)
    dt = time.time() - t0

    # Reset dtype
    _tg._DEFAULT_DTYPE = torch.float32
    _lvg._DEFAULT_DTYPE = torch.float32
    return mean_tm, per_target, dt


def analyze(
    usalign_scores: dict[str, float],
    gpu_scores: dict[str, float],
    label: str,
) -> pd.DataFrame:
    targets = sorted(set(usalign_scores) & set(gpu_scores))
    rows = []
    for tid in targets:
        u = usalign_scores[tid]
        g = gpu_scores[tid]
        rows.append({
            "target": tid,
            "usalign": u,
            "gpu": g,
            "delta": g - u,
            "rel_delta_pct": 100.0 * (g - u) / max(u, 1e-12),
            "gpu_wins": g > u + 1e-6,
            "usalign_wins": u > g + 1e-6,
        })

    df = pd.DataFrame(rows)
    n = len(df)
    gpu_wins = int(df["gpu_wins"].sum())
    usalign_wins = int(df["usalign_wins"].sum())
    ties = n - gpu_wins - usalign_wins

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Targets compared:  {n}")
    print(f"  GPU wins:          {gpu_wins}")
    print(f"  USalign wins:      {usalign_wins}")
    print(f"  Ties (|Δ|<1e-6):   {ties}")
    print(f"  Mean USalign:      {df['usalign'].mean():.6f}")
    print(f"  Mean GPU:          {df['gpu'].mean():.6f}")
    print(f"  Mean Δ (GPU-U):    {df['delta'].mean():.6f}")
    print(f"  Median Δ:          {df['delta'].median():.6f}")
    print(f"  Max Δ (GPU best):  {df['delta'].max():.6f}")
    print(f"  Min Δ (U best):    {df['delta'].min():.6f}")
    print(f"  Std Δ:             {df['delta'].std():.6f}")

    # Statistical test: paired sign test
    pos = (df["delta"] > 1e-6).sum()
    neg = (df["delta"] < -1e-6).sum()
    from scipy.stats import binomtest, wilcoxon
    if pos + neg > 0:
        binom = binomtest(int(pos), int(pos + neg), 0.5, alternative="greater")
        print(f"  Sign test (GPU>U): p={binom.pvalue:.4e} (pos={pos}, neg={neg})")
    if (df["delta"].abs() > 1e-8).sum() >= 5:
        try:
            w_stat, w_p = wilcoxon(
                df["delta"].values, zero_method="wilcox",
                alternative="greater",
            )
            print(f"  Wilcoxon signed-rank (GPU>U): W={w_stat:.1f}, p={w_p:.4e}")
        except Exception:
            pass

    # Per-target details
    print(f"\n  {'Target':<16s} {'USalign':>10s} {'GPU':>10s} {'Δ':>10s} {'%':>8s}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for _, r in df.sort_values("delta", ascending=False).iterrows():
        marker = ""
        if r["gpu_wins"]:
            marker = " <<"
        elif r["usalign_wins"]:
            marker = " >>"
        print(
            f"  {r['target']:<16s} {r['usalign']:10.6f} {r['gpu']:10.6f} "
            f"{r['delta']:+10.6f} {r['rel_delta_pct']:+7.2f}%{marker}"
        )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU vs USalign comparison")
    parser.add_argument("submission", type=str)
    parser.add_argument("--validation", type=str, default=None)
    parser.add_argument("--usalign", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--skip-usalign", action="store_true",
        help="Skip USalign run (use cached scores from a previous run)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    validation_path = (
        Path(args.validation) if args.validation
        else project_root / "stanford-rna-3d-folding-2" / "validation_labels.csv"
    )
    usalign_bin = args.usalign or str(Path(__file__).parent / "USalign")
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    import os
    workers = args.workers or min(os.cpu_count() or 1, 28)

    submission = pd.read_csv(args.submission)
    validation = pd.read_csv(validation_path)

    print(f"Device: {dev}")
    print(f"USalign binary: {usalign_bin}")
    print(f"Targets: {submission['ID'].str.rsplit('_', n=1).str[0].nunique()}")

    # --- Run USalign ---
    if not args.skip_usalign:
        print("\n--- Running USalign ---")
        u_mean, u_scores, u_dt = run_usalign_scoring(
            validation, submission, usalign_bin, workers,
        )
        print(f"USalign mean={u_mean:.6f} time={u_dt:.1f}s")
    else:
        u_scores = {}
        print("\n--- USalign SKIPPED ---")

    # --- Run GPU strict float32 (dp_iter=0) ---
    print("\n--- Running GPU strict float32 dp_iter=0 ---")
    g32_mean, g32_scores, g32_dt = run_gpu_scoring(
        validation, submission, dev, "strict", use_float64=False, dp_iter=0,
    )
    print(f"GPU f32 strict dp0 mean={g32_mean:.6f} time={g32_dt:.1f}s")

    # --- Run GPU strict float64 (dp_iter=0) ---
    print("\n--- Running GPU strict float64 dp_iter=0 ---")
    g64_mean, g64_scores, g64_dt = run_gpu_scoring(
        validation, submission, dev, "strict", use_float64=True, dp_iter=0,
    )
    print(f"GPU f64 strict dp0 mean={g64_mean:.6f} time={g64_dt:.1f}s")

    # --- Run GPU strict float64 (dp_iter=1) — matches USalign's DP alignment ---
    print("\n--- Running GPU strict float64 dp_iter=1 ---")
    g64dp1_mean, g64dp1_scores, g64dp1_dt = run_gpu_scoring(
        validation, submission, dev, "strict", use_float64=True, dp_iter=1,
    )
    print(f"GPU f64 strict dp1 mean={g64dp1_mean:.6f} time={g64dp1_dt:.1f}s")

    # --- Analysis ---
    if u_scores:
        analyze(u_scores, g32_scores, "GPU f32 strict dp0 vs USalign")
        analyze(u_scores, g64_scores, "GPU f64 strict dp0 vs USalign")
        analyze(u_scores, g64dp1_scores, "GPU f64 strict dp1 vs USalign")

        # Internal consistency
        analyze(g32_scores, g64_scores, "GPU f64 vs f32 (precision gap, dp0)")
        analyze(g64_scores, g64dp1_scores, "GPU f64 dp1 vs dp0 (DP alignment effect)")

        # Summary
        print(f"\n{'='*70}")
        print(f"  TIMING SUMMARY")
        print(f"{'='*70}")
        print(f"  USalign:        {u_dt:7.1f}s")
        print(f"  GPU f32 dp0:    {g32_dt:7.1f}s  ({u_dt/g32_dt:.1f}x)")
        print(f"  GPU f64 dp0:    {g64_dt:7.1f}s  ({u_dt/g64_dt:.1f}x)")
        print(f"  GPU f64 dp1:    {g64dp1_dt:7.1f}s  ({u_dt/g64dp1_dt:.1f}x)")
    else:
        analyze(g32_scores, g64_scores, "GPU f64 vs f32 (precision gap, dp0)")


if __name__ == "__main__":
    main()
