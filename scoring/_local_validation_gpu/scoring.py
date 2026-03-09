"""Scoring functions for GPU TM-score validation."""

from __future__ import annotations

from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import itertools

import numpy as np
import pandas as pd
import torch

from . import runtime as rt


def _score_pair_batches(
    pred_coords: torch.Tensor,
    pred_valid: torch.Tensor,
    native_coords: torch.Tensor,
    native_valid: torch.Tensor,
    pair_chunk: int = 256,
    max_iter: int | None = None,
    use_fragment_search: bool = rt.DEFAULT_USE_FRAGMENT_SEARCH,
    max_mem_gb: float = rt.DEFAULT_MAX_MEM_GB,
    dp_iter: int = rt.DEFAULT_DP_ITER,
) -> float:
    """Score max TM across all pred/native frame pairs."""
    dtype = pred_coords.dtype
    P = pred_coords.shape[0]
    F = native_coords.shape[0]
    if P == 0 or F == 0:
        return 0.0

    total_pairs = P * F
    best = 0.0

    for start in range(0, total_pairs, pair_chunk):
        end = min(start + pair_chunk, total_pairs)
        pair_ids = torch.arange(start, end, dtype=torch.long, device=pred_coords.device)
        p_idx = torch.div(pair_ids, F, rounding_mode="floor")
        n_idx = torch.remainder(pair_ids, F)

        p = pred_coords.index_select(0, p_idx)
        n = native_coords.index_select(0, n_idx)
        nv = native_valid.index_select(0, n_idx)
        v = pred_valid.index_select(0, p_idx) & nv
        Lnorm = nv.sum(dim=1).to(dtype=dtype)

        scores = torch.zeros((end - start,), dtype=dtype, device=pred_coords.device)
        good = Lnorm > 2.0
        if torch.any(good):
            d0, d0_search, score_d8 = rt.d0_from_length(Lnorm[good], mol="rna", dtype=dtype)
            scores_good = rt.tmscore_search(
                pred=p[good],
                native=n[good],
                valid=v[good],
                d0=d0,
                d0_search=d0_search,
                score_d8=score_d8,
                Lnorm=Lnorm[good],
                max_iter=max_iter,
                use_fragment_search=use_fragment_search,
                max_mem_gb=max_mem_gb,
                dp_iter=dp_iter,
            )
            scores[good] = scores_good

        chunk_best = float(scores.max().item()) if scores.numel() > 0 else 0.0
        if chunk_best > best:
            best = chunk_best
    return best


@torch.inference_mode()
def score_target(
    target_id: str,
    group_native: pd.DataFrame,
    group_predicted: pd.DataFrame,
    usalign_bin: Optional[str] = None,
    device: Optional[str] = None,
    max_iter: int | None = None,
    use_fragment_search: bool = rt.DEFAULT_USE_FRAGMENT_SEARCH,
    max_mem_gb: float = rt.DEFAULT_MAX_MEM_GB,
    dp_iter: int = rt.DEFAULT_DP_ITER,
) -> float:
    """Score a single target.

    Signature intentionally mirrors local_validation_mt.score_target.
    `usalign_bin` is accepted for API compatibility but ignored.
    """
    _ = usalign_bin
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = rt._DEFAULT_DTYPE

    gn = rt._sorted_by_resid(group_native)
    gp = rt._sorted_by_resid(group_predicted)
    if len(gn) != len(gp):
        raise ValueError(f"Residue length mismatch for {target_id}: native={len(gn)}, pred={len(gp)}")

    native_ids = rt._available_frames(gn, rt.MAX_NATIVE_FRAMES)
    pred_ids = rt._available_frames(gp, rt.MAX_PRED_FRAMES)
    if not pred_ids:
        return 0.0

    native_coords, native_valid = rt.extract_coords(gn, native_ids, dev, dtype=dtype)
    pred_coords, pred_valid = rt.extract_coords(gp, pred_ids, dev, dtype=dtype)

    if native_coords.shape[0] == 0:
        raise ValueError(f"No native models with coordinates for target {target_id}")

    # Keep only native frames with at least one resolved residue.
    keep_native = native_valid.any(dim=1)
    if not torch.any(keep_native):
        raise ValueError(f"No native models with coordinates for target {target_id}")
    native_coords = native_coords[keep_native]
    native_valid = native_valid[keep_native]

    has_chain_copy = ("chain" in gn.columns) and ("copy" in gn.columns)
    is_multicopy = has_chain_copy and (float(gn["copy"].astype(float).max()) > 1.0)

    if not is_multicopy:
        return _score_pair_batches(
            pred_coords,
            pred_valid,
            native_coords,
            native_valid,
            pair_chunk=256,
            max_iter=max_iter,
            use_fragment_search=use_fragment_search,
            max_mem_gb=max_mem_gb,
            dp_iter=dp_iter,
        )

    group_blocks, chain_labels = rt._multimer_group_indices(gn)
    n_blocks = len(group_blocks)
    if n_blocks <= 1:
        return _score_pair_batches(
            pred_coords,
            pred_valid,
            native_coords,
            native_valid,
            pair_chunk=256,
            max_iter=max_iter,
            use_fragment_search=use_fragment_search,
            max_mem_gb=max_mem_gb,
            dp_iter=dp_iter,
        )

    if n_blocks > 4:
        return rt._hungarian_chain_score(
            pred_coords, pred_valid, native_coords, native_valid,
            group_blocks, chain_labels, max_iter, use_fragment_search,
            max_mem_gb=max_mem_gb, dp_iter=dp_iter,
        )

    # <= 4 groups: enumerate all permutations (<= 24)
    best = 0.0
    for perm in itertools.permutations(range(n_blocks)):
        perm_idx = np.concatenate([group_blocks[i] for i in perm], axis=0)
        idx_t = torch.from_numpy(perm_idx).to(device=pred_coords.device, dtype=torch.long)
        p_perm = pred_coords.index_select(1, idx_t)
        v_perm = pred_valid.index_select(1, idx_t)
        s = _score_pair_batches(
            p_perm,
            v_perm,
            native_coords,
            native_valid,
            pair_chunk=256,
            max_iter=max_iter,
            use_fragment_search=use_fragment_search,
            max_mem_gb=max_mem_gb,
            dp_iter=dp_iter,
        )
        if s > best:
            best = s
    return best


@torch.inference_mode()
def score_parallel(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    usalign_bin: str,
    workers: int,
    mode: str = "thread",
    device: Optional[str] = None,
    max_iter: int | None = None,
    use_fragment_search: bool = rt.DEFAULT_USE_FRAGMENT_SEARCH,
    max_mem_gb: float = rt.DEFAULT_MAX_MEM_GB,
    dp_iter: int = rt.DEFAULT_DP_ITER,
    exact_rescore_topk: int = rt.DEFAULT_EXACT_RESCORE_TOPK,
) -> tuple[float, dict[str, float]]:
    """Score all targets with cross-target batching for monomers.

    Signature mirrors local_validation_mt.score_parallel. `workers`/`mode` are
    accepted for compatibility; computation runs on a single GPU device.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = rt._DEFAULT_DTYPE

    sol = solution.copy()
    sub = submission.copy()
    sol["target_id"] = sol["ID"].apply(lambda x: "_".join(str(x).split("_")[:-1]))
    sub["target_id"] = sub["ID"].apply(lambda x: "_".join(str(x).split("_")[:-1]))

    targets = list(sol["target_id"].unique())
    groups_native = {tid: sol[sol["target_id"] == tid] for tid in targets}
    groups_pred = {tid: sub[sub["target_id"] == tid] for tid in targets}

    # Classify targets and pre-sort DataFrames
    mono_tids: list[str] = []
    multi_tids: list[str] = []
    target_lengths: dict[str, int] = {}
    sorted_dfs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for tid in targets:
        gn = rt._sorted_by_resid(groups_native[tid])
        gp = rt._sorted_by_resid(groups_pred[tid])
        sorted_dfs[tid] = (gn, gp)
        target_lengths[tid] = len(gn)
        has_cc = ("chain" in gn.columns) and ("copy" in gn.columns)
        if has_cc and float(gn["copy"].astype(float).max()) > 1.0:
            multi_tids.append(tid)
        else:
            mono_tids.append(tid)

    per_target: dict[str, float] = {}

    # Batch-score monomer targets grouped by sequence length
    for group in rt._group_targets_by_length(mono_tids, target_lengths):
        max_n = max(target_lengths[tid] for tid in group)
        all_p: list[torch.Tensor] = []
        all_n: list[torch.Tensor] = []
        all_v: list[torch.Tensor] = []
        all_nv: list[torch.Tensor] = []
        pair_map: list[tuple[str, int, list[int], int]] = []  # (tid, count, pred_ids, native_count)

        for tid in group:
            gn, gp = sorted_dfs[tid]
            native_ids = rt._available_frames(gn, rt.MAX_NATIVE_FRAMES)
            pred_ids = rt._available_frames(gp, rt.MAX_PRED_FRAMES)
            if not pred_ids:
                pair_map.append((tid, 0, [], 0))
                continue

            nc, nv = rt.extract_coords(gn, native_ids, dev, dtype=dtype)
            pc, pv = rt.extract_coords(gp, pred_ids, dev, dtype=dtype)
            if nc.shape[0] == 0:
                pair_map.append((tid, 0, pred_ids, 0))
                continue
            keep = nv.any(dim=1)
            nc, nv = nc[keep], nv[keep]
            if nc.shape[0] == 0:
                pair_map.append((tid, 0, pred_ids, 0))
                continue

            P_cnt, N_len = pc.shape[0], pc.shape[1]
            F_cnt = nc.shape[0]
            pad_n = max_n - N_len

            # Broadcast: (P,1,N,3) x (1,F,N,3) -> (P*F, N, 3)
            p_pairs = pc.unsqueeze(1).expand(P_cnt, F_cnt, N_len, 3).reshape(P_cnt * F_cnt, N_len, 3)
            n_pairs = nc.unsqueeze(0).expand(P_cnt, F_cnt, N_len, 3).reshape(P_cnt * F_cnt, N_len, 3)
            pv_pairs = pv.unsqueeze(1).expand(P_cnt, F_cnt, N_len).reshape(P_cnt * F_cnt, N_len)
            nv_pairs = nv.unsqueeze(0).expand(P_cnt, F_cnt, N_len).reshape(P_cnt * F_cnt, N_len)
            v_pairs = pv_pairs & nv_pairs

            if pad_n > 0:
                p_pairs = torch.nn.functional.pad(p_pairs, (0, 0, 0, pad_n))
                n_pairs = torch.nn.functional.pad(n_pairs, (0, 0, 0, pad_n))
                v_pairs = torch.nn.functional.pad(v_pairs, (0, pad_n), value=False)
                nv_pairs = torch.nn.functional.pad(nv_pairs, (0, pad_n), value=False)

            all_p.append(p_pairs)
            all_n.append(n_pairs)
            all_v.append(v_pairs)
            all_nv.append(nv_pairs)
            pair_map.append((tid, P_cnt * F_cnt, pred_ids, F_cnt))

        if not all_p:
            for tid, _, _, _ in pair_map:
                per_target.setdefault(tid, 0.0)
            continue

        bp = torch.cat(all_p)
        bn = torch.cat(all_n)
        bv = torch.cat(all_v)
        bnv = torch.cat(all_nv)
        total_pairs = bp.shape[0]
        Lnorm = bnv.sum(dim=1).to(dtype=dtype)
        good = Lnorm > 2.0
        scores = torch.zeros(total_pairs, dtype=dtype, device=bp.device)
        if good.any():
            d0, d0_search, score_d8 = rt.d0_from_length(Lnorm[good], dtype=dtype)
            scores[good] = rt.tmscore_search(
                bp[good], bn[good], bv[good],
                d0, d0_search, score_d8, Lnorm[good],
                max_iter=max_iter, use_fragment_search=use_fragment_search,
                max_mem_gb=max_mem_gb, dp_iter=dp_iter,
            )

        offset = 0
        exact_jobs: list[tuple[str, list[int]]] = []
        for tid, count, pred_ids, native_count in pair_map:
            if count > 0:
                score_slice = scores[offset : offset + count]
                per_target[tid] = float(score_slice.max().item())
                if exact_rescore_topk > 0 and usalign_bin and pred_ids and native_count > 0:
                    pred_cnt = min(len(pred_ids), max(1, int(exact_rescore_topk)))
                    per_pred = score_slice.reshape(len(pred_ids), native_count).max(dim=1).values
                    top_idx = torch.topk(per_pred, k=pred_cnt, largest=True).indices.tolist()
                    exact_jobs.append((tid, [pred_ids[i] for i in top_idx]))
            else:
                per_target[tid] = 0.0
            offset += count

        if exact_jobs:
            max_workers = workers if workers and workers > 0 else len(exact_jobs)
            max_workers = max(1, min(max_workers, len(exact_jobs)))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(
                        rt._exact_rescore_predictions,
                        tid,
                        sorted_dfs[tid][0],
                        sorted_dfs[tid][1],
                        pred_ids,
                        usalign_bin,
                        workers,
                    ): tid
                    for tid, pred_ids in exact_jobs
                }
                for fut, tid in [(f, futs[f]) for f in futs]:
                    per_target[tid] = fut.result()

    # Multicopy targets: score individually (need permutation handling)
    for tid in multi_tids:
        if exact_rescore_topk > 0 and usalign_bin:
            from scoring.local_validation_mt import score_target as _score_target_mt

            per_target[tid] = _score_target_mt(
                tid,
                groups_native[tid],
                groups_pred[tid],
                usalign_bin,
            )
        else:
            per_target[tid] = score_target(
                target_id=tid,
                group_native=groups_native[tid],
                group_predicted=groups_pred[tid],
                device=dev,
                max_iter=max_iter,
                use_fragment_search=use_fragment_search,
                max_mem_gb=max_mem_gb,
                dp_iter=dp_iter,
            )

    mean_tm = float(sum(per_target.values()) / len(per_target)) if per_target else 0.0
    return mean_tm, per_target
