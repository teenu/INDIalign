#!/usr/bin/env python3
"""Side-by-side train_labels benchmark for INDIalign search/refinement modes.

This benchmark samples monomer targets from Stanford RNA Folding 2 train_labels,
applies the existing random-rotation plus Gaussian-noise perturbation protocol,
and then scores the same perturbed set with:

- USalign (reference)
- baseline: strict pair-local seed search without TM-weighted refinement
- weighted_all: broadened pair-local search plus TM-weighted refinement on all
  candidate transforms
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import scoring.local_validation_gpu as lvg
import scoring.tmscore_gpu as tg
from scoring.benchmark_500 import generate_perturbed_submission, run_usalign_batch, sample_targets
from scoring.local_validation_gpu import _configure_tmscore_backend, score_parallel


MODE_CONFIGS: dict[str, dict[str, object]] = {
    "baseline": {
        "description": "strict pair-local search without TM-weighted refinement",
        "config": {
            "d0_add_offsets": (),
            "contact_seed_mults": (),
            "contact_seed_add_offsets": (),
            "contact_seed_tol_mult": 0.25,
            "contact_seed_max_anchors": 0,
            "search_uses_score_d8": True,
            "tm_weighted_refine_iters": 0,
            "tm_weighted_refine_topk": 0,
            "tm_weighted_refine_score_margin": 0.0,
        },
    },
    "weighted_all": {
        "description": "broadened pair-local search with TM-weighted refinement on all candidates",
        "config": {
            "d0_add_offsets": (1.0, 2.0, 3.0, 4.0, 5.0),
            "contact_seed_mults": (1.0, 1.5, 2.0),
            "contact_seed_add_offsets": (0.0, 1.0, 2.0, 3.0),
            "contact_seed_tol_mult": 0.25,
            "contact_seed_max_anchors": 8,
            "search_uses_score_d8": False,
            "tm_weighted_refine_iters": 4,
            "tm_weighted_refine_topk": 0,
            "tm_weighted_refine_score_margin": 0.0,
        },
    },
}


def _run_gpu_mode(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    device: str,
    max_mem_gb: float,
    cfg: dict[str, object],
) -> tuple[float, float, dict[str, float]]:
    tg.reset_backend_runtime_state()
    tg._DEFAULT_DTYPE = torch.float64
    lvg._DEFAULT_DTYPE = torch.float64
    _configure_tmscore_backend("strict")
    tg.configure_pair_local_search(**cfg)

    t0 = time.perf_counter()
    mean_tm, per_target = score_parallel(
        solution,
        submission,
        device=device,
        max_iter=None,
        use_fragment_search=True,
        max_mem_gb=max_mem_gb,
        dp_iter=0,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)
    dt = time.perf_counter() - t0
    return mean_tm, dt, per_target


def _compare_scores(ref: dict[str, float], other: dict[str, float]) -> dict[str, float | int]:
    keys = sorted(set(ref) & set(other))
    deltas = np.array([other[k] - ref[k] for k in keys], dtype=np.float64)
    wins = int((deltas > 1e-6).sum())
    losses = int((deltas < -1e-6).sum())
    ties = len(keys) - wins - losses
    return {
        "n": len(keys),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "std_delta": float(deltas.std()),
        "p90_abs_delta": float(np.quantile(np.abs(deltas), 0.9)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "max_gain": float(deltas.max()),
        "max_loss": float(deltas.min()),
    }


def _print_summary_table(
    usalign: dict[str, object] | None,
    mode_results: dict[str, dict[str, object]],
    comparisons: dict[str, dict[str, float | int]],
) -> None:
    print("\nMode                 Mean TM    Mean Δ vs US   Wins  Losses  Ties   Time(s)")
    print("--------------------------------------------------------------------------")
    if usalign is not None:
        print(
            f"{'USalign':<20s} "
            f"{float(usalign['mean_tm']):8.6f} "
            f"{'—':>13s} "
            f"{'—':>6s} {'—':>7s} {'—':>5s} "
            f"{float(usalign['seconds']):8.2f}"
        )
    for mode_name, result in mode_results.items():
        cmp_key = f"{mode_name}_vs_usalign"
        cmp = comparisons.get(cmp_key)
        delta_str = f"{float(cmp['mean_delta']):+13.6f}" if cmp is not None else f"{'—':>13s}"
        wins_str = f"{int(cmp['wins']):6d}" if cmp is not None else f"{'—':>6s}"
        loss_str = f"{int(cmp['losses']):7d}" if cmp is not None else f"{'—':>7s}"
        tie_str = f"{int(cmp['ties']):5d}" if cmp is not None else f"{'—':>5s}"
        print(
            f"{mode_name:<20s} "
            f"{float(result['mean_tm']):8.6f} "
            f"{delta_str} "
            f"{wins_str} {loss_str} {tie_str} "
            f"{float(result['seconds']):8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Number of monomer targets to sample")
    parser.add_argument("--seed", type=int, default=42, help="Sampling and perturbation seed")
    parser.add_argument("--device", type=str, default=None, help="Scoring device (default: cuda if available)")
    parser.add_argument("--workers", type=int, default=28, help="USalign worker count")
    parser.add_argument("--max-mem-gb", type=float, default=8.0, help="GPU chunking budget in GB")
    parser.add_argument("--usalign", type=str, default=None, help="Path to USalign binary")
    parser.add_argument("--train-labels", type=str, default=None, help="Path to train_labels.csv")
    parser.add_argument("--skip-usalign", action="store_true", help="Skip the USalign reference run")
    parser.add_argument(
        "--out",
        type=str,
        default="/tmp/indialign_train_mode_benchmark.json",
        help="Path to write structured JSON results",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    train_path = (
        Path(args.train_labels) if args.train_labels
        else project_root / "stanford-rna-3d-folding-2" / "train_labels.csv"
    )
    usalign_bin = args.usalign or str(Path(__file__).parent / "USalign")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading train_labels from {train_path}...")
    train = pd.read_csv(train_path, low_memory=False)

    print(f"Sampling {args.n} monomers with seed {args.seed}...")
    targets = sample_targets(train, n=args.n, seed=args.seed)
    print(f"Selected {len(targets)} targets")

    print("Generating perturbed predictions...")
    solution, submission, sigma_map = generate_perturbed_submission(train, targets, seed=args.seed)
    sigmas = np.array(list(sigma_map.values()), dtype=np.float64)
    print(f"Solution rows: {len(solution)} | Submission rows: {len(submission)}")
    print(f"Noise σ range: {sigmas.min():.1f} – {sigmas.max():.1f} Å (mean {sigmas.mean():.1f})")

    results: dict[str, object] = {
        "meta": {
            "train_labels": str(train_path),
            "n_targets": len(targets),
            "seed": int(args.seed),
            "device": device,
            "max_mem_gb": float(args.max_mem_gb),
        },
        "targets": targets,
        "sigma_map": sigma_map,
        "modes": {},
        "comparisons": {},
    }

    usalign_result: dict[str, object] | None = None
    if not args.skip_usalign:
        print(f"\n--- Running USalign ({len(targets)} targets, {args.workers} workers) ---")
        t0 = time.perf_counter()
        u_scores = run_usalign_batch(solution, submission, targets, usalign_bin, workers=args.workers)
        u_dt = time.perf_counter() - t0
        usalign_result = {
            "mean_tm": float(np.mean(list(u_scores.values()))),
            "seconds": float(u_dt),
            "per_target": u_scores,
        }
        results["usalign"] = usalign_result
        print(f"USalign mean TM: {usalign_result['mean_tm']:.6f} | time: {u_dt:.2f}s")

    mode_results: dict[str, dict[str, object]] = {}
    for mode_name, meta in MODE_CONFIGS.items():
        print(f"\n--- Running {mode_name} ---")
        mean_tm, dt, per_target = _run_gpu_mode(
            solution,
            submission,
            device=device,
            max_mem_gb=float(args.max_mem_gb),
            cfg=meta["config"],  # type: ignore[arg-type]
        )
        mode_results[mode_name] = {
            "description": meta["description"],
            "config": meta["config"],
            "mean_tm": float(mean_tm),
            "seconds": float(dt),
            "per_target": per_target,
        }
        print(f"{mode_name} mean TM: {mean_tm:.6f} | time: {dt:.2f}s")

    results["modes"] = mode_results

    comparisons: dict[str, dict[str, float | int]] = {}
    if usalign_result is not None:
        u_scores = usalign_result["per_target"]  # type: ignore[assignment]
        for mode_name, result in mode_results.items():
            comparisons[f"{mode_name}_vs_usalign"] = _compare_scores(
                u_scores,  # type: ignore[arg-type]
                result["per_target"],  # type: ignore[arg-type]
            )
    comparisons["weighted_all_vs_baseline"] = _compare_scores(
        mode_results["baseline"]["per_target"],  # type: ignore[arg-type]
        mode_results["weighted_all"]["per_target"],  # type: ignore[arg-type]
    )
    results["comparisons"] = comparisons

    _print_summary_table(usalign_result, mode_results, comparisons)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
