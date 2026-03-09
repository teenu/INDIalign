#!/usr/bin/env python3
"""Benchmark: GPU TM-score vs USalign on 500 perturbed structure pairs.

Generates predictions by applying random rotation + Gaussian noise to native
coordinates from train_labels.csv. Noise σ is sampled per-target from
Uniform(1, 12) Å to produce a diverse range of TM-scores (0.05–0.95).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation


def sample_targets(
    train_labels: pd.DataFrame, n: int = 500, seed: int = 42,
    min_res: int = 50, max_res: int = 300,
) -> list[str]:
    """Sample n monomer targets with residue count in [min_res, max_res]."""
    df = train_labels.copy()
    df["target_id"] = df["ID"].str.rsplit("_", n=1).str[0]
    sizes = df.groupby("target_id").size()
    copies = df.groupby("target_id")["copy"].max()
    ok = sizes[(sizes >= min_res) & (sizes <= max_res) & (copies <= 1)]
    rng = np.random.RandomState(seed)
    chosen = rng.choice(ok.index.values, size=min(n, len(ok)), replace=False)
    return sorted(chosen.tolist())


def generate_perturbed_submission(
    train_labels: pd.DataFrame,
    targets: list[str],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Generate (solution_df, submission_df, sigma_map).

    solution_df: native coords in validation_labels format (x_1, y_1, z_1).
    submission_df: perturbed coords (random rotation + Gaussian noise).
    sigma_map: {target_id: noise_sigma} for analysis.
    """
    rng = np.random.RandomState(seed)
    df = train_labels.copy()
    df["target_id"] = df["ID"].str.rsplit("_", n=1).str[0]

    sol_rows = []
    sub_rows = []
    sigma_map = {}

    for tid in targets:
        sub_df = df[df["target_id"] == tid].copy()
        sub_df = sub_df.sort_values("resid").reset_index(drop=True)

        coords = sub_df[["x_1", "y_1", "z_1"]].values.astype(np.float64)
        valid = np.all(coords > -1e6, axis=1)
        if valid.sum() < 10:
            continue

        # Random noise σ ~ Uniform(1, 12) Å
        sigma = rng.uniform(1.0, 12.0)
        sigma_map[tid] = sigma

        # Random rotation
        rot = Rotation.random(random_state=rng)
        rotated = rot.apply(coords)

        # Add Gaussian noise
        noise = rng.randn(*rotated.shape) * sigma
        perturbed = rotated + noise
        # Invalidate same residues as native
        perturbed[~valid] = -1e6

        for i, (_, row) in enumerate(sub_df.iterrows()):
            base = {
                "ID": row["ID"],
                "resname": row["resname"],
                "resid": row["resid"],
            }
            sol_row = {**base, "x_1": coords[i, 0], "y_1": coords[i, 1], "z_1": coords[i, 2]}
            if "chain" in sub_df.columns:
                sol_row["chain"] = row["chain"]
            if "copy" in sub_df.columns:
                sol_row["copy"] = row["copy"]
            sol_rows.append(sol_row)

            sub_row = {**base, "x_1": perturbed[i, 0], "y_1": perturbed[i, 1], "z_1": perturbed[i, 2]}
            sub_rows.append(sub_row)

    sol_df = pd.DataFrame(sol_rows)
    sub_df = pd.DataFrame(sub_rows)
    return sol_df, sub_df, sigma_map


def write_pdb(coords: np.ndarray, valid: np.ndarray, path: str) -> int:
    """Write C1' PDB file. Returns count of resolved atoms."""
    n = 0
    with open(path, "w") as f:
        for i in range(len(coords)):
            if not valid[i]:
                continue
            n += 1
            x, y, z = coords[i]
            f.write(
                f"ATOM  {n:>5d}  C1'   A A{i+1:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n"
            )
    return n


def run_usalign_pair(native_pdb: str, pred_pdb: str, usalign_bin: str) -> float:
    """Run USalign on a single pair and return TM2 (normalized by native)."""
    cmd = [usalign_bin, pred_pdb, native_pdb, "-atom", " C1'", "-TMscore", "1"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        matches = re.findall(r"TM-score=\s+([\d.]+)", out.stdout)
        if len(matches) >= 2:
            return float(matches[1])
    except Exception:
        pass
    return 0.0


def run_usalign_batch(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    targets: list[str],
    usalign_bin: str,
    workers: int = 28,
) -> dict[str, float]:
    """Score all targets with USalign (parallel)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    sol = solution.copy()
    sub = submission.copy()
    sol["target_id"] = sol["ID"].str.rsplit("_", n=1).str[0]
    sub["target_id"] = sub["ID"].str.rsplit("_", n=1).str[0]

    results: dict[str, float] = {}

    def _score_one(tid: str) -> tuple[str, float]:
        gn = sol[sol["target_id"] == tid].sort_values("resid").reset_index(drop=True)
        gp = sub[sub["target_id"] == tid].sort_values("resid").reset_index(drop=True)
        native_coords = gn[["x_1", "y_1", "z_1"]].values
        pred_coords = gp[["x_1", "y_1", "z_1"]].values
        native_valid = np.all(native_coords > -1e6, axis=1)
        pred_valid = np.all(pred_coords > -1e6, axis=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            nat_path = os.path.join(tmpdir, "native.pdb")
            pred_path = os.path.join(tmpdir, "pred.pdb")
            write_pdb(native_coords, native_valid, nat_path)
            write_pdb(pred_coords, pred_valid, pred_path)
            tm = run_usalign_pair(nat_path, pred_path, usalign_bin)
        return tid, tm

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_score_one, tid): tid for tid in targets}
        for fut in as_completed(futs):
            tid, tm = fut.result()
            results[tid] = tm

    return results


def run_gpu_batch(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    device: str,
    use_float64: bool = True,
) -> dict[str, float]:
    """Score all targets with GPU scorer."""
    import scoring.tmscore_gpu as _tg
    import scoring.local_validation_gpu as _lvg
    from scoring.local_validation_gpu import _configure_tmscore_backend, score_parallel

    if use_float64:
        _tg._DEFAULT_DTYPE = torch.float64
        _lvg._DEFAULT_DTYPE = torch.float64
    else:
        _tg._DEFAULT_DTYPE = torch.float32
        _lvg._DEFAULT_DTYPE = torch.float32

    _configure_tmscore_backend("strict")

    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)
    _, per_target = score_parallel(
        solution, submission,
        usalign_bin="", workers=0, device=device,
        max_iter=None, use_fragment_search=True,
        max_mem_gb=20.0, dp_iter=0,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)

    _tg._DEFAULT_DTYPE = torch.float32
    _lvg._DEFAULT_DTYPE = torch.float32
    return per_target


def analyze(
    u_scores: dict[str, float],
    g_scores: dict[str, float],
    sigma_map: dict[str, float],
    label: str,
) -> pd.DataFrame:
    targets = sorted(set(u_scores) & set(g_scores) & set(sigma_map))
    rows = []
    for tid in targets:
        u, g, s = u_scores[tid], g_scores[tid], sigma_map[tid]
        rows.append({
            "target": tid, "sigma": s, "usalign": u, "gpu": g,
            "delta": g - u,
            "gpu_wins": g > u + 1e-6,
            "usalign_wins": u > g + 1e-6,
        })

    df = pd.DataFrame(rows)
    n = len(df)
    gpu_wins = int(df["gpu_wins"].sum())
    usalign_wins = int(df["usalign_wins"].sum())
    ties = n - gpu_wins - usalign_wins

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  Pairs:   {n}")
    print(f"  GPU wins:      {gpu_wins}  ({100*gpu_wins/n:.1f}%)")
    print(f"  USalign wins:  {usalign_wins}  ({100*usalign_wins/n:.1f}%)")
    print(f"  Ties:          {ties}  ({100*ties/n:.1f}%)")
    print(f"  Mean Δ:        {df['delta'].mean():.6f}")
    print(f"  Median Δ:      {df['delta'].median():.6f}")
    print(f"  Std Δ:         {df['delta'].std():.6f}")

    # Statistical tests
    from scipy.stats import binomtest, wilcoxon
    nontie = df[df["delta"].abs() > 1e-6]
    pos = (nontie["delta"] > 0).sum()
    neg = (nontie["delta"] < 0).sum()
    if pos + neg > 0:
        bt = binomtest(int(pos), int(pos + neg), 0.5, alternative="greater")
        print(f"  Sign test (GPU>U):    p={bt.pvalue:.4e}  ({pos} vs {neg})")
    if len(nontie) >= 10:
        try:
            w, wp = wilcoxon(nontie["delta"].values, zero_method="wilcox", alternative="greater")
            print(f"  Wilcoxon (GPU>U):     W={w:.0f}, p={wp:.4e}")
        except Exception:
            pass

    # Stratified analysis by TM-score difficulty
    df["tm_avg"] = (df["usalign"] + df["gpu"]) / 2
    bins = [(0, 0.2, "hard (TM<0.2)"), (0.2, 0.4, "medium (0.2-0.4)"),
            (0.4, 0.6, "moderate (0.4-0.6)"), (0.6, 1.01, "easy (TM>0.6)")]
    print(f"\n  {'Stratum':<22s} {'N':>5s} {'GPU>':>5s} {'U>':>5s} {'Tie':>5s} {'Mean Δ':>10s}")
    print(f"  {'-'*22} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*10}")
    for lo, hi, name in bins:
        s = df[(df["tm_avg"] >= lo) & (df["tm_avg"] < hi)]
        if len(s) == 0:
            continue
        gw = int(s["gpu_wins"].sum())
        uw = int(s["usalign_wins"].sum())
        ti = len(s) - gw - uw
        print(f"  {name:<22s} {len(s):5d} {gw:5d} {uw:5d} {ti:5d} {s['delta'].mean():+10.6f}")

    # Top GPU wins and losses
    print(f"\n  Top 10 GPU wins:")
    for _, r in df.nlargest(10, "delta").iterrows():
        print(f"    {r['target']:<12s} σ={r['sigma']:5.1f}  U={r['usalign']:.4f}  G={r['gpu']:.4f}  Δ={r['delta']:+.4f}")
    print(f"\n  Top 10 USalign wins:")
    for _, r in df.nsmallest(10, "delta").iterrows():
        print(f"    {r['target']:<12s} σ={r['sigma']:5.1f}  U={r['usalign']:.4f}  G={r['gpu']:.4f}  Δ={r['delta']:+.4f}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--usalign", type=str, default=None)
    parser.add_argument("--train-labels", type=str, default=None)
    parser.add_argument("--skip-usalign", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    train_path = (
        Path(args.train_labels) if args.train_labels
        else project_root / "stanford-rna-3d-folding-2" / "train_labels.csv"
    )
    usalign_bin = args.usalign or str(Path(__file__).parent / "USalign")
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading train_labels from {train_path}...")
    train = pd.read_csv(train_path, low_memory=False)

    print(f"Sampling {args.n} monomers...")
    targets = sample_targets(train, n=args.n, seed=args.seed)
    print(f"Selected {len(targets)} targets")

    print("Generating perturbed predictions...")
    solution, submission, sigma_map = generate_perturbed_submission(
        train, targets, seed=args.seed,
    )
    print(f"Solution: {len(solution)} rows, Submission: {len(submission)} rows")
    sigmas = np.array(list(sigma_map.values()))
    print(f"Noise σ range: {sigmas.min():.1f} – {sigmas.max():.1f} Å (mean {sigmas.mean():.1f})")

    # USalign
    if not args.skip_usalign:
        print(f"\n--- Running USalign ({len(targets)} targets, {args.workers} workers) ---")
        t0 = time.time()
        u_scores = run_usalign_batch(
            solution, submission, targets, usalign_bin, workers=args.workers,
        )
        u_dt = time.time() - t0
        u_mean = np.mean(list(u_scores.values()))
        print(f"USalign: mean={u_mean:.6f}, time={u_dt:.1f}s")
    else:
        u_scores = {}

    # GPU f64
    print(f"\n--- Running GPU f64 strict ({len(targets)} targets) ---")
    t0 = time.time()
    g_scores = run_gpu_batch(solution, submission, dev, use_float64=True)
    g_dt = time.time() - t0
    g_mean = np.mean(list(g_scores.values()))
    print(f"GPU f64: mean={g_mean:.6f}, time={g_dt:.1f}s")

    # GPU f32
    print(f"\n--- Running GPU f32 strict ({len(targets)} targets) ---")
    t0 = time.time()
    g32_scores = run_gpu_batch(solution, submission, dev, use_float64=False)
    g32_dt = time.time() - t0
    g32_mean = np.mean(list(g32_scores.values()))
    print(f"GPU f32: mean={g32_mean:.6f}, time={g32_dt:.1f}s")

    # Analysis
    if u_scores:
        analyze(u_scores, g_scores, sigma_map, f"GPU f64 vs USalign ({len(targets)} pairs)")
        analyze(u_scores, g32_scores, sigma_map, f"GPU f32 vs USalign ({len(targets)} pairs)")
        analyze(g32_scores, g_scores, sigma_map, f"GPU f64 vs f32 ({len(targets)} pairs)")

        print(f"\n{'='*72}")
        print(f"  TIMING")
        print(f"{'='*72}")
        print(f"  USalign:   {u_dt:7.1f}s")
        print(f"  GPU f64:   {g_dt:7.1f}s  ({u_dt/g_dt:.1f}x)")
        print(f"  GPU f32:   {g32_dt:7.1f}s  ({u_dt/g32_dt:.1f}x)")
    else:
        analyze(g32_scores, g_scores, sigma_map, f"GPU f64 vs f32 ({len(targets)} pairs)")


if __name__ == "__main__":
    main()
