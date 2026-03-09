"""Multimer handling and chain permutation scoring."""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from . import runtime as rt


def _exact_rescore_predictions(
    target_id: str,
    group_native: pd.DataFrame,
    group_predicted: pd.DataFrame,
    pred_ids: list[int],
    usalign_bin: str,
    workers: int = 0,
) -> float:
    """Exact USalign rescore for a selected subset of submission conformers."""
    if not pred_ids or not usalign_bin:
        return 0.0

    from scoring import local_validation_mt as _mt

    has_chain_copy = ("chain" in group_native.columns) and ("copy" in group_native.columns)
    is_multicopy = has_chain_copy and (group_native["copy"].astype(float).max() > 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        native_with_coords: list[int] = []
        for native_cnt in range(1, rt.MAX_NATIVE_FRAMES + 1):
            native_pdb = os.path.join(tmpdir, f"native_{target_id}_{native_cnt}.pdb")
            resolved_native = _mt.write2pdb(group_native, native_cnt, native_pdb)
            if resolved_native > 0:
                native_with_coords.append(native_cnt)
            elif os.path.exists(native_pdb):
                os.remove(native_pdb)

        if not native_with_coords:
            return 0.0

        max_workers = workers if workers and workers > 0 else len(pred_ids)
        max_workers = max(1, min(max_workers, len(pred_ids)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(
                    _mt._score_one_prediction,
                    pred_cnt,
                    is_multicopy,
                    group_predicted,
                    group_native,
                    native_with_coords,
                    tmpdir,
                    usalign_bin,
                    target_id,
                )
                for pred_cnt in pred_ids
            ]
            scores = [f.result() for f in futs]

    return max(scores) if scores else 0.0


def _multimer_group_indices(
    group_native_sorted: pd.DataFrame,
) -> tuple[list[np.ndarray], list[str]]:
    """Return (blocks, chain_labels) for each (chain, copy) group.

    blocks: list of index arrays (one per group).
    chain_labels: chain name for each group (used for per-chain-type permutation).
    """
    groups = list(zip(group_native_sorted["chain"].astype(str), group_native_sorted["copy"].astype(int)))
    by_group: dict[tuple[str, int], list[int]] = {}
    for idx, g in enumerate(groups):
        by_group.setdefault(g, []).append(idx)
    blocks = [np.asarray(v, dtype=np.int64) for v in by_group.values()]
    chain_labels = [k[0] for k in by_group.keys()]
    return blocks, chain_labels


def _group_targets_by_length(
    targets: list[str], target_lengths: dict[str, int], max_ratio: float = 2.0,
) -> list[list[str]]:
    """Group targets by residue count so max/min <= max_ratio within each group."""
    items = sorted(targets, key=lambda tid: target_lengths[tid])
    groups: list[list[str]] = []
    group: list[str] = []
    group_min = 0
    for tid in items:
        n = target_lengths[tid]
        if group and n > group_min * max_ratio:
            groups.append(group)
            group = []
        if not group:
            group_min = n
        group.append(tid)
    if group:
        groups.append(group)
    return groups


def _icp_chain_candidates(
    p_full: torch.Tensor,
    n_full: torch.Tensor,
    pv_full: torch.Tensor,
    nv_full: torch.Tensor,
    block_idx: list[torch.Tensor],
    gis: list[int],
    n_ch: int,
    max_icp_rounds: int = 20,
) -> list[tuple]:
    """Generate candidate permutations via ICP on chain centroids.

    Uses rotation-invariant inter-centroid distance profiles to seed ICP,
    then refines with Kabsch on centroids + Hungarian assignment.
    Returns deduplicated list of permutation tuples.
    """
    from scipy.optimize import linear_sum_assignment

    dtype = p_full.dtype

    # Compute centroids (valid-atom weighted)
    pc = torch.zeros(n_ch, 3, dtype=dtype)
    nc = torch.zeros(n_ch, 3, dtype=dtype)
    for ci in range(n_ch):
        bi = block_idx[gis[ci]]
        pv_mask = pv_full[bi]
        nv_mask = nv_full[bi]
        if pv_mask.any():
            pc[ci] = p_full[bi][pv_mask].mean(dim=0)
        if nv_mask.any():
            nc[ci] = n_full[bi][nv_mask].mean(dim=0)

    # Center both sets
    pc_c = pc - pc.mean(dim=0)
    nc_c = nc - nc.mean(dim=0)

    # Seed 1: distance-profile matching (rotation-invariant)
    D_P = torch.cdist(pc_c.unsqueeze(0), pc_c.unsqueeze(0)).squeeze(0)
    D_N = torch.cdist(nc_c.unsqueeze(0), nc_c.unsqueeze(0)).squeeze(0)
    P_prof = torch.sort(D_P, dim=1).values[:, 1:]
    N_prof = torch.sort(D_N, dim=1).values[:, 1:]
    profile_cost = torch.cdist(P_prof, N_prof).cpu().numpy()
    _, seed_dp = linear_sum_assignment(profile_cost)

    # Seeds: distance-profile, identity, + random restarts
    # ICP on n centroids is O(n^2) per round -- negligible cost.
    rng = np.random.RandomState(0)
    n_random = max(0, 8 * n_ch - 2)  # scale with chain count
    random_seeds = [rng.permutation(n_ch) for _ in range(n_random)]
    seeds = [seed_dp, np.arange(n_ch)] + random_seeds

    found: set[tuple] = set()
    for seed in seeds:
        current = seed.copy()
        visited: set[tuple] = set()
        for _ in range(max_icp_rounds):
            key = tuple(current)
            if key in visited:
                break
            visited.add(key)
            # Kabsch on centroids with current matching
            matched = nc_c[current]
            H = pc_c.T @ matched
            U, S, Vh = torch.linalg.svd(H)
            d = torch.det(Vh.T @ U.T)
            diag = torch.ones(3, dtype=dtype)
            diag[2] = d.sign()
            R = Vh.T @ torch.diag(diag) @ U.T
            rotated = pc_c @ R.T
            # Hungarian on aligned centroid distances
            cdist = torch.cdist(
                rotated.unsqueeze(0), nc_c.unsqueeze(0)
            ).squeeze(0).cpu().numpy()
            _, new_perm = linear_sum_assignment(cdist)
            current = new_perm
        found.add(tuple(current))

    # Always include identity
    found.add(tuple(range(n_ch)))
    # Invert: ICP finds pred->native mapping, but caller needs reorder indices
    return [tuple(int(x) for x in np.argsort(p)) for p in found]


def _hungarian_chain_score(
    pred_coords: torch.Tensor,
    pred_valid: torch.Tensor,
    native_coords: torch.Tensor,
    native_valid: torch.Tensor,
    group_blocks: list[np.ndarray],
    chain_labels: list[str],
    max_iter: int | None,
    use_fragment_search: bool,
    max_mem_gb: float = rt.DEFAULT_MAX_MEM_GB,
    dp_iter: int = rt.DEFAULT_DP_ITER,
) -> float:
    """Score multicopy target using per-chain-type permutation search.

    For each (pred, native) frame pair:
    1. Compute per-chain-pair RMSD cost (independent Kabsch per pair).
    2. Non-degenerate chain types: Hungarian gives a single best assignment.
    3. Degenerate types (identical copies): enumerate all permutations.
    4. Score the cross-product of candidate permutations, take max.
    """
    import math
    from collections import defaultdict
    from itertools import permutations as _perms, product as _product
    from scipy.optimize import linear_sum_assignment

    DEGEN_EPS = 1e-4
    MAX_CANDIDATES = 720  # cap per frame-pair to prevent blowup

    G = len(group_blocks)
    P = pred_coords.shape[0]
    F = native_coords.shape[0]
    device = pred_coords.device
    block_idx = [torch.from_numpy(b).to(device=device, dtype=torch.long)
                 for b in group_blocks]

    chain_to_gis: dict[str, list[int]] = defaultdict(list)
    for gi, chain in enumerate(chain_labels):
        chain_to_gis[chain].append(gi)

    perm_pred_list: list[torch.Tensor] = []
    perm_valid_list: list[torch.Tensor] = []
    native_list: list[torch.Tensor] = []
    native_valid_list: list[torch.Tensor] = []

    for pi in range(P):
        for ni in range(F):
            p_full = pred_coords[pi]
            n_full = native_coords[ni]
            pv_full = pred_valid[pi]
            nv_full = native_valid[ni]

            # Per chain type: cost matrix + candidate permutations
            type_candidates: dict[str, list[tuple]] = {}
            for chain, gis in chain_to_gis.items():
                n_ch = len(gis)
                if n_ch <= 1:
                    type_candidates[chain] = [(0,)]
                    continue

                # Independent per-chain Kabsch RMSD cost matrix
                cost = np.zeros((n_ch, n_ch), dtype=np.float64)
                for ci, gi in enumerate(gis):
                    for cj, gj in enumerate(gis):
                        bi, bj = block_idx[gi], block_idx[gj]
                        vij = pv_full[bi] & nv_full[bj]
                        nv_cnt = vij.sum().item()
                        if nv_cnt < 3:
                            cost[ci, cj] = 1e12
                            continue
                        p_ch = p_full[bi].unsqueeze(0)
                        n_ch_c = n_full[bj].unsqueeze(0)
                        R_ch, t_ch = rt.kabsch_batch(
                            p_ch, n_ch_c, vij.unsqueeze(0))
                        aligned = rt._apply_transform(
                            p_ch, R_ch, t_ch).squeeze(0)
                        d2 = ((aligned - n_full[bj]) ** 2).sum(dim=1)
                        cost[ci, cj] = (
                            (d2 * vij.to(dtype=d2.dtype)).sum().item()
                            / nv_cnt)

                # Degeneracy = within-row variation is negligible.
                # Identical-shape copies have nearly flat rows (column
                # assignment is noise), even if between-row costs differ.
                max_row_range = 0.0
                for r in range(n_ch):
                    row_f = cost[r][cost[r] < 1e11]
                    if len(row_f) > 1:
                        max_row_range = max(max_row_range,
                                            row_f.max() - row_f.min())
                is_degen = max_row_range < DEGEN_EPS

                if is_degen and math.factorial(n_ch) <= MAX_CANDIDATES:
                    type_candidates[chain] = list(_perms(range(n_ch)))
                elif is_degen:
                    # Too many to enumerate. Use ICP on chain centroids
                    # to find candidate permutations.
                    type_candidates[chain] = _icp_chain_candidates(
                        p_full, n_full, pv_full, nv_full,
                        block_idx, gis, n_ch,
                    )
                else:
                    _, col_ind = linear_sum_assignment(cost)
                    inv_col = np.argsort(col_ind)
                    type_candidates[chain] = [tuple(inv_col)]

            # Cap total cross-product to MAX_CANDIDATES
            chain_order = sorted(type_candidates.keys())
            total = 1
            for ch in chain_order:
                total *= len(type_candidates[ch])
            while total > MAX_CANDIDATES:
                biggest = max(chain_order,
                              key=lambda ch: len(type_candidates[ch]))
                # Collapse to identity (degenerate Hungarian = identity)
                type_candidates[biggest] = [
                    tuple(range(len(chain_to_gis[biggest])))]
                total = 1
                for ch in chain_order:
                    total *= len(type_candidates[ch])

            # Cross-product of per-type candidates -> full permutations
            for combo in _product(
                *[type_candidates[ch] for ch in chain_order]
            ):
                perm = list(range(G))
                for idx, chain in enumerate(chain_order):
                    gis = chain_to_gis[chain]
                    tp = combo[idx]
                    for ci, gi in enumerate(gis):
                        perm[gi] = gis[tp[ci]]
                perm_idx = np.concatenate(
                    [group_blocks[perm[j]] for j in range(G)])
                idx_t = torch.from_numpy(perm_idx).to(
                    device=device, dtype=torch.long)
                perm_pred_list.append(
                    pred_coords[pi].index_select(0, idx_t))
                perm_valid_list.append(
                    pred_valid[pi].index_select(0, idx_t))
                native_list.append(native_coords[ni])
                native_valid_list.append(native_valid[ni])

    if not perm_pred_list:
        return 0.0

    # Score all candidates in one batch
    all_pred = torch.stack(perm_pred_list)
    all_pv = torch.stack(perm_valid_list)
    all_native = torch.stack(native_list)
    all_nv = torch.stack(native_valid_list)

    v = all_pv & all_nv
    dtype = all_pred.dtype
    Lnorm = all_nv.sum(dim=1).to(dtype=dtype)
    good = Lnorm > 2.0
    if not good.any():
        return 0.0

    d0, d0_search, score_d8 = rt.d0_from_length(Lnorm[good], dtype=dtype)
    scores = torch.zeros(len(perm_pred_list), dtype=dtype, device=device)
    scores[good] = rt.tmscore_search(
        all_pred[good], all_native[good], v[good],
        d0, d0_search, score_d8, Lnorm[good],
        max_iter=max_iter, use_fragment_search=use_fragment_search,
        max_mem_gb=max_mem_gb, dp_iter=dp_iter,
    )
    return float(scores.max().item())
