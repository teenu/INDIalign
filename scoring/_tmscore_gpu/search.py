"""Seed-based TM-score search and evaluation."""

from __future__ import annotations

import torch

import scoring._tmscore_gpu.runtime as rt


def _fragment_lengths(max_len: int) -> list[int]:
    if max_len < 4:
        return []
    base = int(max_len)
    vals = [
        base,
        base // 2,
        base // 4,
        base // 8,
        64,
        32,
        16,
        8,
        4,
    ]
    vals = sorted({v for v in vals if 4 <= v <= base}, reverse=True)
    return vals


def _start_positions(n: int, frag: int, max_starts: int = 20) -> list[int]:
    total = n - frag + 1
    if total <= 0:
        return []
    if total <= max_starts:
        return list(range(total))
    step = max(1, total // max_starts)
    starts = list(range(0, total, step))
    if starts[-1] != total - 1:
        starts.append(total - 1)
    return starts


def _build_seed_windows(
    N: int, max_lali: int, use_fragment_search: bool, device: torch.device,
) -> torch.Tensor:
    """Build seed window masks for fragment search.

    Returns (K, N) bool tensor. K=1 (global only) when use_fragment_search=False.
    The global seed is an all-True window (ANDed with valid later).
    """
    global_win = torch.ones(1, N, dtype=torch.bool, device=device)
    if not use_fragment_search:
        return global_win

    ms = rt._MAX_FRAG_STARTS if rt._MAX_FRAG_STARTS is not None else (120 if max_lali <= 500 else 50)
    starts_list: list[int] = []
    frags_list: list[int] = []
    for frag in _fragment_lengths(max_lali):
        for start in _start_positions(N, frag, max_starts=ms):
            starts_list.append(start)
            frags_list.append(frag)

    if not starts_list:
        return global_win

    # Vectorized: (K,) starts/frags → (K, N) masks via broadcasting
    positions = torch.arange(N, device=device)
    starts_t = torch.tensor(starts_list, device=device, dtype=torch.long)
    frags_t = torch.tensor(frags_list, device=device, dtype=torch.long)
    masks = (positions >= starts_t.unsqueeze(1)) & (
        positions < (starts_t + frags_t).unsqueeze(1)
    )
    return torch.cat([global_win, masks], dim=0)


def _build_pairwise_seed_masks(
    valid: torch.Tensor,
    use_fragment_search: bool,
) -> torch.Tensor:
    """Build per-pair fragment seed masks without cross-pair coupling.

    Each pair gets its own fragment-window schedule derived from its own valid
    residue count. The resulting seed dimension is padded to the maximum K in
    the batch; seed semantics need not match across pairs because seeds are
    only ever consumed within the same batch element.
    """
    B, N = valid.shape
    device = valid.device
    if B == 0:
        return torch.zeros((0, 0, N), dtype=torch.bool, device=device)

    valid_counts = valid.sum(dim=1).to(dtype=torch.int64).tolist()
    windows_cache: dict[int, torch.Tensor] = {}
    pair_masks: list[torch.Tensor] = []
    max_k = 0

    for b, local_lali in enumerate(valid_counts):
        windows = windows_cache.get(local_lali)
        if windows is None:
            windows = _build_seed_windows(
                N,
                int(local_lali),
                use_fragment_search,
                device,
            )
            windows_cache[local_lali] = windows
        seed_b = valid[b : b + 1] & windows
        pair_masks.append(seed_b)
        max_k = max(max_k, seed_b.shape[0])

    seed_all = torch.zeros((B, max_k, N), dtype=torch.bool, device=device)
    for b, seed_b in enumerate(pair_masks):
        seed_all[b, : seed_b.shape[0]] = seed_b

    has_enough = (seed_all.sum(dim=2) >= 3).any(dim=0)
    if not has_enough.any():
        return seed_all[:, :0]
    return seed_all[:, has_enough]


def _build_pairwise_topk_seed_masks(
    d2: torch.Tensor,
    valid: torch.Tensor,
    fracs: tuple[float, ...],
) -> torch.Tensor:
    """Build second-pass distance seeds with per-pair top-k budgets."""
    B, N = valid.shape
    if B == 0:
        return torch.zeros((0, len(fracs), N), dtype=torch.bool, device=valid.device)

    valid_counts = valid.sum(dim=1).to(dtype=torch.long)
    min_k = min(3, N)
    rank = None
    seed_masks: list[torch.Tensor] = []

    for frac in fracs:
        k_per = torch.clamp(
            (valid_counts.float() * frac).to(dtype=torch.long),
            min=min_k,
            max=N,
        )
        k_max = int(k_per.max().item())
        if k_max <= 0:
            seed_masks.append(torch.zeros_like(valid))
            continue
        _, topk = d2.topk(k_max, dim=1, largest=False)
        if rank is None or rank.shape[1] != k_max:
            rank = torch.arange(k_max, device=valid.device).unsqueeze(0)
        use_topk = rank < k_per.unsqueeze(1)
        seed = torch.zeros_like(valid)
        seed.scatter_(1, topk, use_topk)
        seed_masks.append(seed & valid)

    return torch.stack(seed_masks, dim=1)


def _build_pairwise_contact_seed_masks(
    d2: torch.Tensor,
    valid: torch.Tensor,
    d0_search: torch.Tensor,
) -> torch.Tensor:
    """Build non-contiguous local-contact seeds from absolute distance radii."""
    radii = rt._contact_seed_radii(d0_search)
    if radii.shape[1] == 0:
        B, N = valid.shape
        return torch.zeros((B, 0, N), dtype=torch.bool, device=valid.device)
    return valid.unsqueeze(1) & (d2.unsqueeze(1) <= (radii.unsqueeze(2) ** 2))


def _build_pairwise_anchor_contact_seed_masks(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    d0_search: torch.Tensor,
) -> torch.Tensor:
    """Build transform-independent local-contact seeds from anchor neighborhoods."""
    B, N, _ = pred.shape
    device = pred.device
    radii = rt._contact_seed_radii(d0_search)
    if B == 0 or radii.shape[1] == 0 or rt._CONTACT_SEED_MAX_ANCHORS <= 0:
        return torch.zeros((B, 0, N), dtype=torch.bool, device=device)

    pair_masks: list[torch.Tensor] = []
    max_k = 0

    for b in range(B):
        valid_b = valid[b]
        if not bool(valid_b.any()):
            pair_masks.append(torch.zeros((0, N), dtype=torch.bool, device=device))
            continue

        pred_b = pred[b : b + 1]
        native_b = native[b : b + 1]
        pd = torch.cdist(pred_b, pred_b).squeeze(0)
        nd = torch.cdist(native_b, native_b).squeeze(0)
        vp = valid_b.unsqueeze(0) & valid_b.unsqueeze(1)
        seeds_b: list[torch.Tensor] = []

        for radius in radii[b]:
            rad = float(radius.item())
            if rad <= 0.0:
                continue
            tol = max(0.5, rad * rt._CONTACT_SEED_TOL_MULT)
            local = (pd <= rad) & (nd <= rad) & vp
            consistent = (pd - nd).abs() <= tol
            masks = local & consistent
            counts = masks.sum(dim=1)
            good = valid_b & (counts >= 3)
            if not bool(good.any()):
                continue

            order = torch.argsort(counts, descending=True)
            picked = 0
            for anchor_idx in order.tolist():
                if not bool(good[anchor_idx]):
                    continue
                seed = masks[anchor_idx] & valid_b
                if int(seed.sum().item()) < 3:
                    continue
                duplicate = any(torch.equal(seed, prev) for prev in seeds_b)
                if duplicate:
                    continue
                seeds_b.append(seed)
                picked += 1
                if picked >= rt._CONTACT_SEED_MAX_ANCHORS:
                    break

        if seeds_b:
            seed_bank_b = torch.stack(seeds_b, dim=0)
        else:
            seed_bank_b = torch.zeros((0, N), dtype=torch.bool, device=device)
        pair_masks.append(seed_bank_b)
        max_k = max(max_k, seed_bank_b.shape[0])

    out = torch.zeros((B, max_k, N), dtype=torch.bool, device=device)
    for b, seed_bank_b in enumerate(pair_masks):
        out[b, : seed_bank_b.shape[0]] = seed_bank_b
    return out


def _evaluate_seed_bank(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    seed_bank: torch.Tensor,
    d0_candidates: torch.Tensor,
    d0: torch.Tensor,
    score_d8: torch.Tensor,
    Lnorm: torch.Tensor,
    max_iter: int,
    max_mem_gb: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Search a bank of seed masks against a bank of pair-local search radii."""
    B, N, _ = pred.shape
    device = pred.device
    dtype = pred.dtype
    K = seed_bank.shape[1]
    C = d0_candidates.shape[1]

    if K == 0 or C == 0:
        eye3 = torch.eye(3, dtype=dtype, device=device)
        return (
            eye3.unsqueeze(0).expand(B, 3, 3).clone(),
            torch.zeros(B, 3, dtype=dtype, device=device),
            torch.full((B,), -torch.inf, dtype=dtype, device=device),
        )

    search_score_d8 = score_d8 if rt._SEARCH_SELECTION_USES_SCORE_D8 else None
    seed_all = seed_bank.unsqueeze(2).expand(B, K, C, N).reshape(B, K * C, N)
    KM = seed_all.shape[1]
    chunk_b, chunk_km = rt._pairseed_chunk_plan(B, N, KM, dtype, max_mem_gb)

    eye3 = torch.eye(3, dtype=dtype, device=device)
    best_R = eye3.unsqueeze(0).expand(B, 3, 3).clone()
    best_t = torch.zeros(B, 3, dtype=dtype, device=device)
    best_sc = torch.full((B,), -torch.inf, dtype=dtype, device=device)

    for start in range(0, B, chunk_b):
        end = min(start + chunk_b, B)
        cb = end - start
        bidx = torch.arange(cb, device=device)

        p = pred[start:end]
        n = native[start:end]
        v = valid[start:end]
        sa = seed_all[start:end]

        d0s_full = d0_candidates[start:end].unsqueeze(1).expand(cb, K, C).reshape(cb, KM)
        ln_full = Lnorm[start:end].unsqueeze(1).expand(cb, KM)
        d0_full = d0[start:end].unsqueeze(1).expand(cb, KM)
        sd8_full = (
            search_score_d8[start:end].unsqueeze(1).expand(cb, KM)
            if search_score_d8 is not None
            else None
        )

        best_sc_chunk = torch.full((cb,), -torch.inf, dtype=dtype, device=device)
        best_R_chunk = eye3.unsqueeze(0).expand(cb, 3, 3).clone()
        best_t_chunk = torch.zeros(cb, 3, dtype=dtype, device=device)

        for k0 in range(0, KM, chunk_km):
            k1 = min(k0 + chunk_km, KM)
            ks = k1 - k0

            sa_k = sa[:, k0:k1]
            p_exp = p.unsqueeze(1).expand(cb, ks, N, 3).reshape(cb * ks, N, 3)
            n_exp = n.unsqueeze(1).expand(cb, ks, N, 3).reshape(cb * ks, N, 3)
            v_exp = v.unsqueeze(1).expand(cb, ks, N).reshape(cb * ks, N)
            s_flat = sa_k.reshape(cb * ks, N)

            R_all, t_all = _iterative_seed_refine(
                p_exp,
                n_exp,
                v_exp,
                s_flat,
                d0s_full[:, k0:k1].reshape(cb * ks),
                max_iter,
            )
            scores = None
            if rt._TM_WEIGHTED_REFINE_ITERS > 0:
                if rt._tm_weighted_refine_refines_all(ks) and rt._TM_WEIGHTED_REFINE_SCORE_MARGIN <= 0.0:
                    R_all, t_all, scores = rt._tm_weighted_refine(
                        p_exp,
                        n_exp,
                        v_exp,
                        R_all,
                        t_all,
                        d0_full[:, k0:k1].reshape(cb * ks),
                        ln_full[:, k0:k1].reshape(cb * ks),
                        score_d8=sd8_full[:, k0:k1].reshape(cb * ks) if sd8_full is not None else None,
                        max_iter=rt._TM_WEIGHTED_REFINE_ITERS,
                        max_mem_gb=max_mem_gb,
                    )
                else:
                    scores = rt._tm_score_fused(
                        p_exp, n_exp, v_exp, R_all, t_all,
                        d0_full[:, k0:k1].reshape(cb * ks),
                        ln_full[:, k0:k1].reshape(cb * ks),
                        sd8_full[:, k0:k1].reshape(cb * ks) if sd8_full is not None else None,
                    )
                    scores_2d = scores.reshape(cb, ks)
                    refine_mask_2d = rt._weighted_refine_candidate_mask(scores_2d)
                    if bool(refine_mask_2d.any()):
                        refine_idx = torch.nonzero(refine_mask_2d.reshape(cb * ks), as_tuple=False).squeeze(1)
                        r_ref, t_ref, sc_ref = rt._tm_weighted_refine(
                            p_exp.index_select(0, refine_idx),
                            n_exp.index_select(0, refine_idx),
                            v_exp.index_select(0, refine_idx),
                            R_all.index_select(0, refine_idx),
                            t_all.index_select(0, refine_idx),
                            d0_full[:, k0:k1].reshape(cb * ks).index_select(0, refine_idx),
                            ln_full[:, k0:k1].reshape(cb * ks).index_select(0, refine_idx),
                            score_d8=(
                                sd8_full[:, k0:k1].reshape(cb * ks).index_select(0, refine_idx)
                                if sd8_full is not None
                                else None
                            ),
                            max_iter=rt._TM_WEIGHTED_REFINE_ITERS,
                            max_mem_gb=max_mem_gb,
                        )
                        R_all[refine_idx] = r_ref
                        t_all[refine_idx] = t_ref
                        scores[refine_idx] = sc_ref
            if scores is None:
                scores = rt._tm_score_fused(
                    p_exp, n_exp, v_exp, R_all, t_all,
                    d0_full[:, k0:k1].reshape(cb * ks),
                    ln_full[:, k0:k1].reshape(cb * ks),
                    sd8_full[:, k0:k1].reshape(cb * ks) if sd8_full is not None else None,
                )
            scores_2d = scores.reshape(cb, ks)
            best_local_sc, best_local_idx = scores_2d.max(dim=1)
            better_local = best_local_sc > best_sc_chunk
            if better_local.any():
                R_cands = R_all.reshape(cb, ks, 3, 3)[bidx, best_local_idx]
                t_cands = t_all.reshape(cb, ks, 3)[bidx, best_local_idx]
                best_R_chunk[better_local] = R_cands[better_local]
                best_t_chunk[better_local] = t_cands[better_local]
                best_sc_chunk[better_local] = best_local_sc[better_local]

        best_R[start:end] = best_R_chunk
        best_t[start:end] = best_t_chunk
        best_sc[start:end] = best_sc_chunk

    return best_R, best_t, best_sc


def _iterative_seed_refine(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    seed_mask: torch.Tensor,
    d0_search: torch.Tensor,
    max_iter: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    work = seed_mask & valid
    B = pred.shape[0]
    dtype = pred.dtype
    device = pred.device

    use_triton_refine = rt._can_use_triton_refine(pred, native, valid) and d0_search.is_contiguous()
    if use_triton_refine:
        valid_u8 = valid if valid.dtype == torch.uint8 else valid.to(dtype=torch.uint8)
        work_mask = work.to(dtype=torch.uint8)
        d0_search_sq = d0_search * d0_search
        sel_u8 = torch.empty_like(valid_u8)
        near_flags = torch.empty((B,), dtype=torch.int32, device=device)
        R, t = rt.kabsch_batch(pred, native, work_mask)
    else:
        valid_u8 = None
        d0_search_sq = None
        sel_u8 = None
        near_flags = None
        R, t = rt.kabsch_batch(pred, native, work)

    # Run fixed iterations — no convergence checks.  For large batches (the
    # common case), torch.equal(sel, prev) almost never fires because
    # individual elements converge at different rates.  Meanwhile each check
    # forces a CPU-GPU sync (~5 ms) that stalls the pipeline, dominating
    # wall-clock time.  The torch.where below already handles degenerate
    # elements (sel_good=False) by keeping R, t unchanged.
    for _ in range(max_iter):
        if use_triton_refine:
            if rt._TRITON_REFINE_PARITY_GUARD:
                near_flags.zero_()
                sel = rt._select_mask_triton(
                    pred, native, R, t, valid_u8, d0_search_sq, sel_u8, near_flags,
                    abs_tol=rt._TRITON_REFINE_PARITY_ABS_TOL,
                    rel_tol=rt._TRITON_REFINE_PARITY_REL_TOL,
                )
                near_idx = torch.nonzero(near_flags != 0, as_tuple=False).squeeze(1)
                if near_idx.numel() > 0:
                    pred_near = pred.index_select(0, near_idx)
                    native_near = native.index_select(0, near_idx)
                    R_near = R.index_select(0, near_idx)
                    t_near = t.index_select(0, near_idx)
                    moved_near = rt._apply_transform(pred_near, R_near, t_near)
                    dist2_near = ((moved_near - native_near) ** 2).sum(dim=2)
                    valid_near = valid.index_select(0, near_idx)
                    d0s_near = d0_search.index_select(0, near_idx)
                    sel_exact = valid_near & (dist2_near <= (d0s_near[:, None] ** 2))
                    sel[near_idx] = sel_exact.to(dtype=sel.dtype)
            else:
                near_flags.zero_()
                sel = rt._select_mask_triton(pred, native, R, t, valid_u8, d0_search_sq, sel_u8, near_flags)
            sel_good = sel.sum(dim=1) >= 3
        else:
            moved = rt._apply_transform(pred, R, t)
            dist2 = ((moved - native) ** 2).sum(dim=2)
            sel = valid & (dist2 <= (d0_search[:, None] ** 2))
            sel_good = sel.sum(dim=1) >= 3
        R_new, t_new = rt.kabsch_batch(pred, native, sel)
        R = torch.where(sel_good[:, None, None], R_new, R)
        t = torch.where(sel_good[:, None], t_new, t)
    return R, t


def tmscore_search(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    d0: torch.Tensor,
    d0_search: torch.Tensor,
    score_d8: torch.Tensor,
    Lnorm: torch.Tensor,
    max_iter: int | None = None,
    use_fragment_search: bool = True,
    max_mem_gb: float = 20.0,
    dp_iter: int = 0,
    pred_valid: torch.Tensor | None = None,
    native_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched TM-score search with seed-parallel fragment evaluation.

    Fragment seeds are evaluated in large GPU batches (memory-aware chunked
    when needed) rather than iterating over seeds serially. Each seed is also
    tried with multiple d0_search cutoff multipliers to find TM-score-optimal
    superpositions (USalign behaviour).

    When ``dp_iter > 0``, an iterative Needleman-Wunsch DP refinement step
    runs after the seed search, discovering optimal residue correspondences
    (TM-align behaviour).

    Parameters
    ----------
    pred, native:
        Shape (B, N, 3) tensors (both padded to the same N).
    valid:
        Shape (B, N), bool — AND of pred/native validity for same-index ops.
    d0, d0_search, score_d8, Lnorm:
        Shape (B,).
    max_iter:
        Refinement iterations per seed. ``None`` → use ``_MAX_ITER`` global
        (20 by default, 8 when ultrafast backend is configured).
    dp_iter:
        Number of DP-superposition refinement iterations. 0 = no DP (default).
    pred_valid, native_valid:
        Separate validity masks for pred and native. When provided, DP
        refinement uses them independently so that NW can discover offset
        correspondences between structures of different lengths.
    """
    if max_iter is None:
        max_iter = rt._MAX_ITER
    if pred.ndim != 3 or native.ndim != 3 or pred.shape != native.shape:
        raise ValueError("pred and native must be shape (B, N, 3) and match.")
    if valid.shape != pred.shape[:2]:
        raise ValueError("valid mask must be shape (B, N).")

    B, N, _ = pred.shape
    device = pred.device
    dtype = pred.dtype

    if B == 0:
        return torch.zeros((0,), dtype=dtype, device=device)

    # Build pair-local fragment seeds. This avoids coupling a pair's search
    # schedule to unrelated lengths elsewhere in the batch.
    seed_all = _build_pairwise_seed_masks(valid, use_fragment_search)
    anchor_seed_all = _build_pairwise_anchor_contact_seed_masks(pred, native, valid, d0_search)
    if anchor_seed_all.shape[1] > 0:
        seed_all = torch.cat([seed_all, anchor_seed_all], dim=1)
    if seed_all.shape[1] == 0:
        return torch.zeros(B, dtype=dtype, device=device)
    eye3 = torch.eye(3, dtype=dtype, device=device)
    d0_candidates = rt._d0_search_candidates(d0_search)
    best_R, best_t, cur_score = _evaluate_seed_bank(
        pred,
        native,
        valid,
        seed_all,
        d0_candidates,
        d0,
        score_d8,
        Lnorm,
        max_iter,
        max_mem_gb,
    )

    # Two-pass: batch distance-sorted seeds (fracs × mults), chunked to
    # avoid a large coordinate expansion.
    d2 = rt._dist2_fused(pred, native, valid, best_R, best_t)

    fracs = rt._DIST_FRACS
    seeds_3 = _build_pairwise_topk_seed_masks(d2, valid, fracs)
    contact_seeds = _build_pairwise_contact_seed_masks(d2, valid, d0_search)
    if rt._tm_weighted_refine_is_selective() and contact_seeds.shape[1] > 0:
        for seed_bank in (seeds_3, contact_seeds):
            if seed_bank.shape[1] == 0:
                continue
            tp_R, tp_t, tp_score = _evaluate_seed_bank(
                pred,
                native,
                valid,
                seed_bank,
                d0_candidates,
                d0,
                score_d8,
                Lnorm,
                max_iter,
                max_mem_gb,
            )
            cur_score = rt._merge_best_transforms(best_R, best_t, cur_score, tp_R, tp_t, tp_score)
    else:
        seeds_tp = torch.cat([seeds_3, contact_seeds], dim=1) if contact_seeds.shape[1] > 0 else seeds_3
        if seeds_tp.shape[1] > 0:
            tp_R, tp_t, tp_score = _evaluate_seed_bank(
                pred,
                native,
                valid,
                seeds_tp,
                d0_candidates,
                d0,
                score_d8,
                Lnorm,
                max_iter,
                max_mem_gb,
            )
            cur_score = rt._merge_best_transforms(best_R, best_t, cur_score, tp_R, tp_t, tp_score)

    # DP refinement: iterative NW alignment discovery
    if dp_iter > 0:
        dp_R, dp_t, dp_score = rt._dp_refine(
            pred, native, valid, best_R, best_t, d0, score_d8, Lnorm, dp_iter,
            pred_valid=pred_valid, native_valid=native_valid,
        )
        # Identity score with DP-refined superposition
        dp_id_score = rt._tm_score_fused(pred, native, valid, dp_R, dp_t, d0, Lnorm)
        # Identity score with original (pre-DP) superposition
        orig_score = rt._tm_score_fused(pred, native, valid, best_R, best_t, d0, Lnorm)
        # Best of DP-aligned, DP-identity, and original-identity
        final_score = torch.max(torch.max(dp_score, dp_id_score), orig_score)

        # When separate masks are provided, also try DP from identity
        # superposition.  The seed search uses same-index pairs and cannot
        # discover offset correspondences; starting DP from R=I, t=0 lets
        # the score matrix reveal the true cross-index signal.
        if pred_valid is not None and native_valid is not None:
            id_R = eye3.unsqueeze(0).expand(B, 3, 3).clone()
            id_t = torch.zeros(B, 3, dtype=dtype, device=device)
            id_dp_R, id_dp_t, id_dp_score = rt._dp_refine(
                pred, native, valid, id_R, id_t, d0, score_d8, Lnorm,
                dp_iter, pred_valid=pred_valid, native_valid=native_valid,
            )
            id_dp_id_score = rt._tm_score_fused(
                pred, native, valid, id_dp_R, id_dp_t, d0, Lnorm,
            )
            final_score = torch.max(
                final_score,
                torch.max(id_dp_score, id_dp_id_score),
            )
    else:
        final_score = rt._tm_score_fused(pred, native, valid, best_R, best_t, d0, Lnorm)

    final_score = torch.where(Lnorm > 2.0, final_score, torch.zeros_like(final_score))
    return final_score
