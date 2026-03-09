"""Dynamic programming alignment refinement."""
from __future__ import annotations

import torch

from . import runtime as rt


def _score_matrix(
    moved: torch.Tensor,
    native: torch.Tensor,
    pred_valid: torch.Tensor,
    nat_valid: torch.Tensor,
    d0: torch.Tensor,
    score_d8: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise TM-score similarity matrix. Returns (B, Np, Nn)."""
    p2 = (moved ** 2).sum(dim=2)
    n2 = (native ** 2).sum(dim=2)
    cross = torch.bmm(moved, native.transpose(1, 2))
    d2 = torch.clamp(p2.unsqueeze(2) + n2.unsqueeze(1) - 2.0 * cross, min=0.0)
    d0_sq = torch.clamp(d0[:, None, None] ** 2, min=1e-12)
    score = 1.0 / (1.0 + d2 / d0_sq)
    if score_d8 is not None:
        score = score * (d2 <= (score_d8[:, None, None] ** 2)).to(dtype=score.dtype)
    mask = pred_valid.unsqueeze(2) & nat_valid.unsqueeze(1)
    return score * mask.to(dtype=score.dtype)


def _nw_dp(score_mat: torch.Tensor) -> torch.Tensor:
    """Anti-diagonal wavefront Needleman-Wunsch with zero gap penalty.

    Returns trace matrix (B, Np+1, Nn+1) int8: 0=diag, 1=up, 2=left.
    """
    B, Np, Nn = score_mat.shape
    device = score_mat.device
    dtype = score_mat.dtype
    H = torch.zeros(B, Np + 1, Nn + 1, dtype=dtype, device=device)
    trace = torch.zeros(B, Np + 1, Nn + 1, dtype=torch.int8, device=device)
    for d in range(2, Np + Nn + 2):
        i_lo = max(1, d - Nn)
        i_hi = min(Np, d - 1)
        if i_lo > i_hi:
            continue
        ii = torch.arange(i_lo, i_hi + 1, device=device)
        jj = d - ii
        diag = H[:, ii - 1, jj - 1] + score_mat[:, ii - 1, jj - 1]
        up = H[:, ii - 1, jj]
        left = H[:, ii, jj - 1]
        cands = torch.stack([diag, up, left], dim=2)
        best_val, best_dir = cands.max(dim=2)
        H[:, ii, jj] = best_val
        trace[:, ii, jj] = best_dir.to(torch.int8)
    return trace


def _nw_traceback(
    trace: torch.Tensor, Np: int, Nn: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Traceback NW alignment paths from trace matrix (fully GPU-native).

    Returns (align_p, align_n, n_aligned):
      align_p, align_n: (B, max_pairs) int64, 0-indexed residue indices, -1 = padding.
      n_aligned: (B,) int64.
    """
    B = trace.shape[0]
    device = trace.device
    max_pairs = min(Np, Nn)

    bi = torch.arange(B, device=device)
    i = torch.full((B,), Np, dtype=torch.long, device=device)
    j = torch.full((B,), Nn, dtype=torch.long, device=device)

    match_i: list[torch.Tensor] = []
    match_j: list[torch.Tensor] = []
    match_mask: list[torch.Tensor] = []
    active = (i > 0) & (j > 0)
    for _ in range(Np + Nn):
        if not active.any():
            break
        dirs = trace[bi, i, j]
        is_diag = (dirs == 0) & active
        is_up = (dirs == 1) & active
        is_left = (~is_diag & ~is_up) & active
        match_i.append(i - 1)
        match_j.append(j - 1)
        match_mask.append(is_diag)
        i = i - (is_diag | is_up).long()
        j = j - (is_diag | is_left).long()
        active = (i > 0) & (j > 0)

    if not match_i:
        return (
            torch.full((B, max_pairs), -1, dtype=torch.long, device=device),
            torch.full((B, max_pairs), -1, dtype=torch.long, device=device),
            torch.zeros(B, dtype=torch.long, device=device),
        )

    match_i.reverse()
    match_j.reverse()
    match_mask.reverse()
    all_i = torch.stack(match_i, dim=1)  # (B, steps)
    all_j = torch.stack(match_j, dim=1)
    all_m = torch.stack(match_mask, dim=1)
    n_aligned = all_m.sum(dim=1)

    # Vectorized extraction: cumsum gives output positions for matched pairs
    cumsum = all_m.long().cumsum(dim=1)  # (B, steps)
    out_idx = (cumsum - 1).clamp(min=0)

    align_p = torch.full((B, max_pairs), -1, dtype=torch.long, device=device)
    align_n = torch.full((B, max_pairs), -1, dtype=torch.long, device=device)

    scatter_mask = all_m & (out_idx < max_pairs)
    b_indices = torch.arange(B, device=device).unsqueeze(1).expand_as(all_m)
    align_p[b_indices[scatter_mask], out_idx[scatter_mask]] = all_i[scatter_mask]
    align_n[b_indices[scatter_mask], out_idx[scatter_mask]] = all_j[scatter_mask]

    return align_p, align_n, n_aligned


def _dp_refine_chunk(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    d0: torch.Tensor,
    score_d8: torch.Tensor,
    Lnorm: torch.Tensor,
    max_iter: int = 5,
    pred_valid: torch.Tensor | None = None,
    native_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-chunk DP refinement (no memory chunking)."""
    pv = pred_valid if pred_valid is not None else valid
    nv = native_valid if native_valid is not None else valid
    B, N, _ = pred.shape
    device = pred.device
    dtype = pred.dtype
    cur_R, cur_t = R.clone(), t.clone()
    best_R, best_t = R.clone(), t.clone()
    best_score = torch.zeros(B, dtype=dtype, device=device)
    for _ in range(max_iter):
        moved = rt._apply_transform(pred, cur_R, cur_t)
        smat = _score_matrix(moved, native, pv, nv, d0, score_d8)
        trace = _nw_dp(smat)
        align_p, align_n, n_aligned = _nw_traceback(trace, N, N)
        max_pairs = align_p.shape[1]
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_pairs)
        safe_p = align_p.clamp(min=0)
        safe_n = align_n.clamp(min=0)
        pair_mask = (align_p >= 0) & pv[b_idx, safe_p] & nv[b_idx, safe_n]
        P_aligned = pred[b_idx, safe_p]
        N_aligned = native[b_idx, safe_n]
        has_enough = pair_mask.sum(dim=1) >= 3
        R_new, t_new = rt.kabsch_batch(P_aligned, N_aligned, pair_mask)
        cur_R = torch.where(has_enough[:, None, None], R_new, cur_R)
        cur_t = torch.where(has_enough[:, None], t_new, cur_t)
        moved_new = rt._apply_transform(pred, cur_R, cur_t)
        d2 = ((moved_new[b_idx, safe_p] - N_aligned) ** 2).sum(dim=2)
        term = 1.0 / (1.0 + d2 / torch.clamp(d0[:, None] ** 2, min=1e-12))
        term = term * pair_mask.to(dtype=term.dtype)
        score = term.sum(dim=1) / torch.clamp(Lnorm, min=1.0)
        better = score > best_score
        best_R[better] = cur_R[better]
        best_t[better] = cur_t[better]
        best_score[better] = score[better]
    return best_R, best_t, best_score


def _dp_refine(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    d0: torch.Tensor,
    score_d8: torch.Tensor,
    Lnorm: torch.Tensor,
    max_iter: int = 5,
    pred_valid: torch.Tensor | None = None,
    native_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iterative DP-superposition refinement (TM-align core loop).

    Takes initial (R, t) from seed search and iteratively discovers better
    residue correspondences via Needleman-Wunsch on the TM-score similarity
    matrix. Returns (best_R, best_t, best_score).

    Chunks by batch dimension to avoid OOM on the (B, N, N) score matrix.
    """
    B, N, _ = pred.shape
    dtype = pred.dtype
    # Score matrix is (chunk, N, N) — budget ~2 GB for it.
    itemsize = torch.empty((), dtype=dtype).element_size()
    budget = int(2.0 * (1024 ** 3))
    bytes_per_elem = max(1, N * N * itemsize * 6)  # score_matrix + NW + transients
    chunk_b = max(1, budget // bytes_per_elem)
    if chunk_b >= B:
        return _dp_refine_chunk(
            pred, native, valid, R, t, d0, score_d8, Lnorm,
            max_iter, pred_valid, native_valid,
        )
    # Chunked path
    best_R = R.clone()
    best_t = t.clone()
    best_score = torch.zeros(B, dtype=dtype, device=pred.device)
    for s in range(0, B, chunk_b):
        e = min(s + chunk_b, B)
        pv_c = pred_valid[s:e] if pred_valid is not None else None
        nv_c = native_valid[s:e] if native_valid is not None else None
        cR, ct, cs = _dp_refine_chunk(
            pred[s:e], native[s:e], valid[s:e],
            R[s:e], t[s:e], d0[s:e], score_d8[s:e], Lnorm[s:e],
            max_iter, pv_c, nv_c,
        )
        best_R[s:e] = cR
        best_t[s:e] = ct
        best_score[s:e] = cs
    return best_R, best_t, best_score
