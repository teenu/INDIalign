"""Rigid-body alignment and TM-score computation."""
from __future__ import annotations

import torch

from . import runtime as rt


def _eigh_largest_eigvec_4x4(F: torch.Tensor) -> torch.Tensor:
    B = F.shape[0]
    dtype = F.dtype
    device = F.device

    def _safe_chunk(Fc: torch.Tensor) -> torch.Tensor:
        try:
            _, vecs = torch.linalg.eigh(Fc)
            q = vecs[:, :, -1]
        except torch._C._LinAlgError:
            q = None

        if q is not None:
            # cuSOLVER can silently return NaN for ill-conditioned matrices.
            nan_rows = q.isnan().any(dim=1)
            if not nan_rows.any():
                return q
            # Fall through to CPU fallback for NaN rows only.
            good_rows = ~nan_rows
            if good_rows.any():
                out = torch.zeros(Fc.shape[0], 4, dtype=dtype, device=device)
                out[:, 0] = 1.0
                out[good_rows] = q[good_rows]
                Fc_bad = Fc[nan_rows]
            else:
                out = torch.zeros(Fc.shape[0], 4, dtype=dtype, device=device)
                out[:, 0] = 1.0
                Fc_bad = Fc
        else:
            out = torch.zeros(Fc.shape[0], 4, dtype=dtype, device=device)
            out[:, 0] = 1.0
            nan_rows = torch.ones(Fc.shape[0], dtype=torch.bool, device=device)
            Fc_bad = Fc

        # CPU fallback for failed/NaN rows.
        Fc_cpu = Fc_bad.detach().cpu().to(dtype=torch.float64)
        bad_indices = torch.nonzero(nan_rows, as_tuple=False).squeeze(1)
        for j, i in enumerate(bad_indices.tolist()):
            try:
                _, v = torch.linalg.eigh(Fc_cpu[j:j+1])
                qv = v[0, :, -1]
                if not qv.isnan().any():
                    out[i] = qv.to(dtype=dtype, device=device)
            except Exception:
                pass
        return out

    if B <= rt._EIGH_CHUNK:
        return _safe_chunk(F)

    results = torch.empty(B, 4, dtype=dtype, device=device)
    for start in range(0, B, rt._EIGH_CHUNK):
        end = min(start + rt._EIGH_CHUNK, B)
        results[start:end] = _safe_chunk(F[start:end])
    return results


def _kabsch_finalize_from_cov(
    H: torch.Tensor,
    cP: torch.Tensor,
    cQ: torch.Tensor,
    count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = H.shape[0]
    dtype = H.dtype
    device = H.device

    # Identify degenerate elements BEFORE eigensolve — cuSOLVER crashes on
    # zero/near-zero 4×4 matrices from collinear or too-few atoms.
    too_small = count.squeeze(1) < 3

    Sxx = H[:, 0, 0]; Sxy = H[:, 0, 1]; Sxz = H[:, 0, 2]
    Syx = H[:, 1, 0]; Syy = H[:, 1, 1]; Syz = H[:, 1, 2]
    Szx = H[:, 2, 0]; Szy = H[:, 2, 1]; Szz = H[:, 2, 2]

    F = torch.empty(B, 4, 4, dtype=dtype, device=device)
    F[:, 0, 0] = Sxx + Syy + Szz
    F[:, 0, 1] = Syz - Szy;    F[:, 1, 0] = F[:, 0, 1]
    F[:, 0, 2] = Szx - Sxz;    F[:, 2, 0] = F[:, 0, 2]
    F[:, 0, 3] = Sxy - Syx;    F[:, 3, 0] = F[:, 0, 3]
    F[:, 1, 1] = Sxx - Syy - Szz
    F[:, 1, 2] = Sxy + Syx;    F[:, 2, 1] = F[:, 1, 2]
    F[:, 1, 3] = Szx + Sxz;    F[:, 3, 1] = F[:, 1, 3]
    F[:, 2, 2] = -Sxx + Syy - Szz
    F[:, 2, 3] = Syz + Szy;    F[:, 3, 2] = F[:, 2, 3]
    F[:, 3, 3] = -Sxx - Syy + Szz

    # Identity quaternion for degenerate elements; only eigensolve the rest.
    q = torch.zeros(B, 4, dtype=dtype, device=device)
    q[:, 0] = 1.0  # identity quaternion default
    good = ~too_small
    n_good = good.sum().item()
    if n_good > 0:
        q[good] = _eigh_largest_eigvec_4x4(F[good])

    # cuSOLVER can silently return NaN eigenvectors for ill-conditioned
    # matrices (collinear atoms, near-zero covariance) without raising.
    # Replace any NaN quaternions with identity before building R.
    nan_q = q.isnan().any(dim=1)
    if nan_q.any():
        q[nan_q] = 0.0
        q[nan_q, 0] = 1.0

    q = q / torch.clamp(torch.linalg.vector_norm(q, dim=1, keepdim=True), min=1e-12)

    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty(B, 3, 3, dtype=dtype, device=device)
    R[:, 0, 0] = q0*q0 + q1*q1 - q2*q2 - q3*q3
    R[:, 0, 1] = 2.0*(q1*q2 + q0*q3)
    R[:, 0, 2] = 2.0*(q1*q3 - q0*q2)
    R[:, 1, 0] = 2.0*(q1*q2 - q0*q3)
    R[:, 1, 1] = q0*q0 - q1*q1 + q2*q2 - q3*q3
    R[:, 1, 2] = 2.0*(q2*q3 + q0*q1)
    R[:, 2, 0] = 2.0*(q1*q3 + q0*q2)
    R[:, 2, 1] = 2.0*(q2*q3 - q0*q1)
    R[:, 2, 2] = q0*q0 - q1*q1 - q2*q2 + q3*q3

    t = cQ - torch.bmm(cP.unsqueeze(1), R).squeeze(1)

    # Sanitize degenerate and any remaining NaN elements.
    bad = (count.squeeze(1) < 3) | R.isnan().any(dim=2).any(dim=1) | t.isnan().any(dim=1)
    if bad.any():
        eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(B, 3, 3)
        zero = torch.zeros((B, 3), dtype=dtype, device=device)
        R = torch.where(bad[:, None, None], eye, R)
        t = torch.where(bad[:, None], zero, t)
    return R, t


def _kabsch_batch_torch(P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = P.dtype
    m = mask.to(dtype=dtype)
    w = m.unsqueeze(-1)
    count = m.sum(dim=1, keepdim=True)
    safe_count = torch.clamp(count, min=1.0)

    cP = (P * w).sum(dim=1) / safe_count
    cQ = (Q * w).sum(dim=1) / safe_count

    Pc = P - cP.unsqueeze(1)
    Qc = Q - cQ.unsqueeze(1)
    # Disable TF32 for covariance matmul: TF32 truncates mantissa to 10 bits
    # which can degrade rotation quality for near-planar or small point sets.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    H = torch.bmm((Pc * w).transpose(1, 2), Qc)
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return _kabsch_finalize_from_cov(H, cP, cQ, count)


def _kabsch_batch_weighted(
    P: torch.Tensor,
    Q: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = P.dtype
    w = weights.to(dtype=dtype)
    w3 = w.unsqueeze(-1)
    support = (w > 1e-8).sum(dim=1, keepdim=True).to(dtype=dtype)
    weight_sum = torch.clamp(w.sum(dim=1, keepdim=True), min=1e-12)

    cP = (P * w3).sum(dim=1) / weight_sum
    cQ = (Q * w3).sum(dim=1) / weight_sum

    Pc = P - cP.unsqueeze(1)
    Qc = Q - cQ.unsqueeze(1)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    H = torch.bmm((Pc * w3).transpose(1, 2), Qc)
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return _kabsch_finalize_from_cov(H, cP, cQ, support)


def kabsch_batch(
    P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Kabsch alignment from P -> Q using Horn's quaternion method.

    Backend selection:
      - Triton fused Kabsch (fast path, gated by size/dtype/device checks)
      - Torch Kabsch with cuSOLVER ``eigh`` (default)

    Parameters
    ----------
    P, Q:
        Shape (B, N, 3).
    mask:
        Shape (B, N), bool.
    """
    if P.ndim != 3 or Q.ndim != 3 or P.shape != Q.shape or P.shape[-1] != 3:
        raise ValueError("P and Q must have shape (B, N, 3) and match.")
    if mask.shape != P.shape[:2]:
        raise ValueError("mask must have shape (B, N).")

    if rt._can_use_triton_kabsch(P, Q, mask):
        return rt._kabsch_fused_triton(P, Q, mask)

    return _kabsch_batch_torch(P, Q, mask)


def _apply_transform(P: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.bmm(P, R) + t.unsqueeze(1)


def _tm_score_impl(
    transformed: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    d0: torch.Tensor,
    Lnorm: torch.Tensor,
    score_d8: torch.Tensor | None = None,
) -> torch.Tensor:
    d2 = ((transformed - native) ** 2).sum(dim=2)
    term = 1.0 / (1.0 + d2 / torch.clamp(d0[:, None] ** 2, min=1e-12))
    if score_d8 is not None:
        term = term * (d2 <= (score_d8[:, None] ** 2)).to(dtype=term.dtype)
    term = term * valid.to(dtype=term.dtype)
    denom = torch.clamp(Lnorm, min=1.0)
    return term.sum(dim=1) / denom


def _activate_torch_compile() -> None:
    """Replace rt._tm_score with a torch.compiled version."""
    if rt._tm_score_compiled is None:
        rt._tm_score_compiled = torch.compile(_tm_score_impl, dynamic=True)
    rt._tm_score = rt._tm_score_compiled


def _deactivate_torch_compile() -> None:
    rt._tm_score = _tm_score_impl


def _tm_score_fused(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    d0: torch.Tensor,
    Lnorm: torch.Tensor,
    score_d8: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform + TM-score, fused when Triton is available."""
    if rt._can_use_triton_score(pred) and R.is_contiguous() and t.is_contiguous():
        return rt._tm_score_fused_triton(pred, native, valid, R, t, d0, Lnorm, score_d8)
    moved = _apply_transform(pred, R, t)
    return rt._tm_score(moved, native, valid, d0, Lnorm, score_d8=score_d8)


def _weighted_refine_chunk_size(
    B: int,
    N: int,
    dtype: torch.dtype,
    max_mem_gb: float,
) -> int:
    """Return a conservative sub-batch size for weighted TM refinement."""
    if B <= 0:
        return 1

    if N <= 256:
        base_cap = 8192
    elif N <= 512:
        base_cap = 4096
    elif N <= 1024:
        base_cap = 2048
    else:
        base_cap = 1024

    if dtype == torch.float32:
        base_cap *= 2

    # Weighted Kabsch uses cuSOLVER batched eigensolve internally; that stage
    # scales poorly in workspace size, so keep a wider safety margin than the
    # fragment-seed chunk planner.
    scale = max(0.25, min(float(max_mem_gb) / 20.0, 2.0))
    chunk = int(base_cap * scale)
    chunk = min(chunk, max(256, rt._EIGH_CHUNK // 4))
    return min(B, max(1, chunk))


def _dist2_fused(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Fused transform + distance squared, invalid -> 1e12. Falls back to torch."""
    if rt._can_use_triton_score(pred) and R.is_contiguous() and t.is_contiguous():
        return rt._dist2_fused_triton(pred, native, valid, R, t)
    moved = _apply_transform(pred, R, t)
    d2 = ((moved - native) ** 2).sum(dim=2)
    return d2 + (~valid).to(dtype=d2.dtype) * 1e12


def _tm_weighted_refine(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    d0: torch.Tensor,
    Lnorm: torch.Tensor,
    score_d8: torch.Tensor | None = None,
    max_iter: int = 0,
    max_mem_gb: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refine a transform using TM-style soft weights instead of hard cutoffs."""
    B, N, _ = pred.shape
    chunk = _weighted_refine_chunk_size(B, N, pred.dtype, max_mem_gb)
    if chunk < B:
        out_R = torch.empty_like(R)
        out_t = torch.empty_like(t)
        out_score = torch.empty_like(d0)
        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            r_i, t_i, s_i = _tm_weighted_refine(
                pred[start:end],
                native[start:end],
                valid[start:end],
                R[start:end],
                t[start:end],
                d0[start:end],
                Lnorm[start:end],
                score_d8=score_d8[start:end] if score_d8 is not None else None,
                max_iter=max_iter,
                max_mem_gb=max_mem_gb,
            )
            out_R[start:end] = r_i
            out_t[start:end] = t_i
            out_score[start:end] = s_i
        return out_R, out_t, out_score

    if max_iter <= 0:
        score = _tm_score_fused(pred, native, valid, R, t, d0, Lnorm, score_d8=score_d8)
        return R, t, score

    dtype = pred.dtype
    valid_f = valid.to(dtype=dtype)
    d0_sq = torch.clamp(d0[:, None] ** 2, min=1e-12)

    best_R = R.clone()
    best_t = t.clone()
    best_score = _tm_score_fused(pred, native, valid, best_R, best_t, d0, Lnorm, score_d8=score_d8)
    cur_R = best_R
    cur_t = best_t

    for _ in range(max_iter):
        d2 = _dist2_fused(pred, native, valid, cur_R, cur_t)
        weights = valid_f / torch.clamp(1.0 + d2 / d0_sq, min=1e-12) ** 2
        if score_d8 is not None:
            weights = weights * (d2 <= (score_d8[:, None] ** 2)).to(dtype=dtype)
        support = (weights > 1e-8).sum(dim=1) >= 3
        if not bool(support.any()):
            break

        cand_R, cand_t = _kabsch_batch_weighted(pred, native, weights)
        cand_score = _tm_score_fused(pred, native, valid, cand_R, cand_t, d0, Lnorm, score_d8=score_d8)
        better = support & (cand_score > best_score)
        if better.any():
            best_R[better] = cand_R[better]
            best_t[better] = cand_t[better]
            best_score = torch.where(better, cand_score, best_score)
            cur_R = torch.where(better[:, None, None], cand_R, cur_R)
            cur_t = torch.where(better[:, None], cand_t, cur_t)
        else:
            break

    return best_R, best_t, best_score


def _weighted_refine_candidate_mask(scores_2d: torch.Tensor) -> torch.Tensor:
    """Select a small candidate subset for TM-weighted refinement."""
    cb, ks = scores_2d.shape
    if cb == 0 or ks == 0:
        return torch.zeros_like(scores_2d, dtype=torch.bool)

    if rt._TM_WEIGHTED_REFINE_TOPK <= 0 or rt._TM_WEIGHTED_REFINE_TOPK >= ks:
        mask = torch.ones_like(scores_2d, dtype=torch.bool)
    else:
        topk = min(rt._TM_WEIGHTED_REFINE_TOPK, ks)
        _, top_idx = scores_2d.topk(topk, dim=1)
        mask = torch.zeros_like(scores_2d, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)

    if rt._TM_WEIGHTED_REFINE_SCORE_MARGIN > 0.0:
        thresh = scores_2d.max(dim=1, keepdim=True).values - rt._TM_WEIGHTED_REFINE_SCORE_MARGIN
        mask |= scores_2d >= thresh
    return mask


def _tm_weighted_refine_refines_all(ks: int) -> bool:
    return rt._TM_WEIGHTED_REFINE_TOPK <= 0 or rt._TM_WEIGHTED_REFINE_TOPK >= ks


def _tm_weighted_refine_is_selective() -> bool:
    return rt._TM_WEIGHTED_REFINE_ITERS > 0 and (
        rt._TM_WEIGHTED_REFINE_TOPK > 0 or rt._TM_WEIGHTED_REFINE_SCORE_MARGIN > 0.0
    )


def _merge_best_transforms(
    best_R: torch.Tensor,
    best_t: torch.Tensor,
    best_score: torch.Tensor,
    cand_R: torch.Tensor,
    cand_t: torch.Tensor,
    cand_score: torch.Tensor,
) -> torch.Tensor:
    better = cand_score > best_score
    if better.any():
        best_R[better] = cand_R[better]
        best_t[better] = cand_t[better]
        best_score = torch.where(better, cand_score, best_score)
    return best_score
