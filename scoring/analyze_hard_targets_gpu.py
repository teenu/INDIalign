#!/usr/bin/env python3
"""Diagnose pair-local search basins on hard Stanford validation targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import scoring.tmscore_gpu as tg
import scoring.local_validation_gpu as lvg
from scoring.local_validation_gpu import _configure_tmscore_backend, _available_frames, _sorted_by_resid, extract_coords
from scoring.local_validation_mt import score_target as usalign_score_target


TARGETS = ("9OD4", "9E9Q", "9OBM", "9RVP")
PROTOTYPE_CFG = {
    "d0_add_offsets": (1.0, 2.0, 3.0, 4.0, 5.0),
    "contact_seed_mults": (1.0, 1.5, 2.0),
    "contact_seed_add_offsets": (0.0, 1.0, 2.0, 3.0),
    "contact_seed_tol_mult": 0.25,
    "contact_seed_max_anchors": 8,
    "search_uses_score_d8": False,
    "tm_weighted_refine_iters": 4,
    "tm_weighted_refine_topk": 2,
    "tm_weighted_refine_score_margin": 0.01,
}


def _configure_mode(mode: str) -> None:
    tg.reset_backend_runtime_state()
    tg._DEFAULT_DTYPE = torch.float64
    lvg._DEFAULT_DTYPE = torch.float64
    _configure_tmscore_backend("strict")
    if mode == "baseline":
        tg.configure_pair_local_search(
            d0_add_offsets=(),
            contact_seed_mults=(),
            contact_seed_add_offsets=(),
            search_uses_score_d8=True,
        )
        return
    if mode == "prototype":
        tg.configure_pair_local_search(**PROTOTYPE_CFG)
        return
    raise ValueError(f"Unknown mode: {mode}")


def _single_frame(
    df: pd.DataFrame,
    frame_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    coords, valid = extract_coords(df, [frame_id], device, dtype=torch.float64)
    return coords, valid


def _score_transform(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    d0: torch.Tensor,
    Lnorm: torch.Tensor,
) -> float:
    return float(tg._tm_score_fused(pred, native, valid, R, t, d0, Lnorm).item())


def _analyze_pair(
    pred: torch.Tensor,
    pred_valid: torch.Tensor,
    native: torch.Tensor,
    native_valid: torch.Tensor,
) -> dict[str, float | str]:
    valid = pred_valid & native_valid
    Lnorm = native_valid.sum(dim=1).to(dtype=pred.dtype)
    d0, d0_search, score_d8 = tg.d0_from_length(Lnorm, dtype=pred.dtype)
    d0_candidates = tg._d0_search_candidates(d0_search)

    frag_bank = tg._build_pairwise_seed_masks(valid, use_fragment_search=True)
    anchor_bank = tg._build_pairwise_anchor_contact_seed_masks(pred, native, valid, d0_search)
    seed_bank = torch.cat([frag_bank, anchor_bank], dim=1) if anchor_bank.shape[1] > 0 else frag_bank
    first_R, first_t, first_search = tg._evaluate_seed_bank(
        pred, native, valid, seed_bank, d0_candidates, d0, score_d8, Lnorm, 20, 20.0
    )
    first_final = _score_transform(pred, native, valid, first_R, first_t, d0, Lnorm)

    d2 = tg._dist2_fused(pred, native, valid, first_R, first_t)
    topk_bank = tg._build_pairwise_topk_seed_masks(d2, valid, tg._DIST_FRACS)
    radial_contact_bank = tg._build_pairwise_contact_seed_masks(d2, valid, d0_search)
    union_parts = [topk_bank]
    if radial_contact_bank.shape[1] > 0:
        union_parts.append(radial_contact_bank)
    if anchor_bank.shape[1] > 0:
        union_parts.append(anchor_bank)
    union_bank = torch.cat(union_parts, dim=1) if len(union_parts) > 1 else union_parts[0]

    def _family(bank: torch.Tensor) -> tuple[float, float]:
        if bank.shape[1] == 0:
            return float("-inf"), float("-inf")
        R, t, search_sc = tg._evaluate_seed_bank(
            pred, native, valid, bank, d0_candidates, d0, score_d8, Lnorm, 20, 20.0
        )
        return float(search_sc.item()), _score_transform(pred, native, valid, R, t, d0, Lnorm)

    topk_search, topk_final = _family(topk_bank)
    radial_contact_search, radial_contact_final = _family(radial_contact_bank)
    anchor_search, anchor_final = _family(anchor_bank)
    union_search, union_final = _family(union_bank)

    final_scores = {
        "first_pass_final": first_final,
        "topk_final": topk_final,
        "radial_contact_final": radial_contact_final,
        "anchor_contact_final": anchor_final,
        "union_final": union_final,
    }
    winner = max(final_scores, key=final_scores.get)
    return {
        "first_pass_search": float(first_search.item()),
        "first_pass_final": first_final,
        "topk_search": topk_search,
        "topk_final": topk_final,
        "radial_contact_search": radial_contact_search,
        "radial_contact_final": radial_contact_final,
        "anchor_contact_search": anchor_search,
        "anchor_contact_final": anchor_final,
        "union_search": union_search,
        "union_final": union_final,
        "winner": winner,
        "final_best": final_scores[winner],
    }


def _single_pred_submission(group_pred: pd.DataFrame, pred_idx: int) -> pd.DataFrame:
    gp = group_pred.copy()
    for other in range(1, 6):
        if other == pred_idx:
            continue
        for axis in "xyz":
            gp[f"{axis}_{other}"] = -1e18
    return gp


def analyze_target(
    target_id: str,
    validation: pd.DataFrame,
    submission: pd.DataFrame,
    device: str,
) -> dict:
    gn = _sorted_by_resid(validation[validation["target_id"] == target_id])
    gp = _sorted_by_resid(submission[submission["target_id"] == target_id])
    native_ids = _available_frames(gn, 40)
    pred_ids = _available_frames(gp, 5)

    rows: list[dict] = []
    usalign_by_pred: dict[int, float] = {}
    for pred_idx in pred_ids:
        usalign_by_pred[pred_idx] = usalign_score_target(
            target_id,
            gn,
            _single_pred_submission(gp, pred_idx),
            str(Path(__file__).parent / "USalign"),
        )

    for mode in ("baseline", "prototype"):
        _configure_mode(mode)
        for pred_idx in pred_ids:
            pred_coords, pred_valid = _single_frame(gp, pred_idx, device)
            if not bool(pred_valid.any()):
                continue
            for native_idx in native_ids:
                native_coords, native_valid = _single_frame(gn, native_idx, device)
                if not bool(native_valid.any()):
                    continue
                info = _analyze_pair(pred_coords, pred_valid, native_coords, native_valid)
                info.update(
                    {
                        "target": target_id,
                        "mode": mode,
                        "pred_idx": pred_idx,
                        "native_idx": native_idx,
                        "usalign_pred_score": usalign_by_pred[pred_idx],
                    }
                )
                rows.append(info)

    df = pd.DataFrame(rows)
    best_rows = []
    for mode in ("baseline", "prototype"):
        d = df[df["mode"] == mode]
        per_pred = d.sort_values("final_best", ascending=False).groupby("pred_idx", as_index=False).first()
        target_best = per_pred.sort_values("final_best", ascending=False).iloc[0].to_dict()
        target_best["usalign_best_pred"] = max(usalign_by_pred, key=usalign_by_pred.get)
        target_best["usalign_best_score"] = max(usalign_by_pred.values())
        best_rows.append(target_best)

    return {
        "target": target_id,
        "usalign_by_pred": usalign_by_pred,
        "pair_rows": rows,
        "summary": best_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--targets", nargs="*", default=list(TARGETS))
    parser.add_argument("--out", default="/tmp/indialign_hard_target_diagnostics.json")
    args = parser.parse_args()

    validation = pd.read_csv(args.validation, low_memory=False)
    submission = pd.read_csv(args.submission, low_memory=False)
    validation["target_id"] = validation["ID"].str.rsplit("_", n=1).str[0]
    submission["target_id"] = submission["ID"].str.rsplit("_", n=1).str[0]

    results = []
    for target_id in args.targets:
        print(f"Analyzing {target_id}...", flush=True)
        results.append(analyze_target(target_id, validation, submission, args.device))

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
