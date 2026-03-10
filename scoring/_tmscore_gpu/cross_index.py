"""Cross-index seed construction and rescue refinement."""

from __future__ import annotations

import torch

from . import dp as _dp
from . import runtime as rt


def _local_init_jump(length: int, dense: bool = False) -> int:
    if dense:
        if length > 300:
            jump = 15
        elif length > 200:
            jump = 10
        else:
            jump = 5
        return max(1, min(jump, max(1, length // 4)))
    if length > 250:
        jump = 45
    elif length > 200:
        jump = 35
    elif length > 150:
        jump = 25
    else:
        jump = 15
    return max(1, min(jump, max(1, length // 3)))


def _local_init_fragment_lengths(
    pred_len: int,
    native_len: int,
    dense: bool = False,
) -> list[int]:
    min_len = min(pred_len, native_len)
    if min_len < 4:
        return []
    if dense:
        vals = (
            min_len,
            min(100, min_len // 2),
            min(32, min_len),
            min(24, min_len),
            min(20, min_len // 3),
            16,
            12,
            8,
            4,
        )
    else:
        vals = (
            min(20, min_len // 3),
            min(100, min_len // 2),
        )
    out: list[int] = []
    for val in vals:
        if val >= 4 and val not in out:
            out.append(val)
    return out


def _build_pairwise_local_fragment_starts(
    pred_valid: torch.Tensor,
    native_valid: torch.Tensor,
    dense: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build USalign-style local fragment-pair schedules."""
    B, _ = pred_valid.shape
    device = pred_valid.device
    pred_counts = pred_valid.sum(dim=1).to(dtype=torch.int64).tolist()
    native_counts = native_valid.sum(dim=1).to(dtype=torch.int64).tolist()
    sched_cache: dict[tuple[int, int], tuple[list[int], list[int], list[int]]] = {}
    schedules: list[tuple[list[int], list[int], list[int]]] = []
    max_k = 0

    for pred_len, native_len in zip(pred_counts, native_counts):
        key = (pred_len, native_len)
        sched = sched_cache.get(key)
        if sched is None:
            pred_starts: list[int] = []
            native_starts: list[int] = []
            frag_lens: list[int] = []
            jump_pred = _local_init_jump(pred_len, dense=dense)
            jump_native = _local_init_jump(native_len, dense=dense)
            for frag_len in _local_init_fragment_lengths(pred_len, native_len, dense=dense):
                pred_limit = pred_len - frag_len + 1
                native_limit = native_len - frag_len + 1
                if pred_limit <= 0 or native_limit <= 0:
                    continue
                for pred_start in range(0, pred_limit, jump_pred):
                    for native_start in range(0, native_limit, jump_native):
                        pred_starts.append(pred_start)
                        native_starts.append(native_start)
                        frag_lens.append(frag_len)
            sched = (pred_starts, native_starts, frag_lens)
            sched_cache[key] = sched
        schedules.append(sched)
        max_k = max(max_k, len(sched[0]))

    pred_starts_all = torch.zeros((B, max_k), dtype=torch.long, device=device)
    native_starts_all = torch.zeros((B, max_k), dtype=torch.long, device=device)
    frag_lens_all = torch.zeros((B, max_k), dtype=torch.long, device=device)
    cand_mask = torch.zeros((B, max_k), dtype=torch.bool, device=device)

    for b, (pred_starts, native_starts, frag_lens) in enumerate(schedules):
        if not frag_lens:
            continue
        count = len(frag_lens)
        pred_starts_all[b, :count] = torch.tensor(pred_starts, dtype=torch.long, device=device)
        native_starts_all[b, :count] = torch.tensor(native_starts, dtype=torch.long, device=device)
        frag_lens_all[b, :count] = torch.tensor(frag_lens, dtype=torch.long, device=device)
        cand_mask[b, :count] = True

    return pred_starts_all, native_starts_all, frag_lens_all, cand_mask


def _build_pairwise_thread_starts(
    pred_valid: torch.Tensor,
    native_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build all gapless threading offsets with at least three aligned residues."""
    B, _ = pred_valid.shape
    device = pred_valid.device
    pred_counts = pred_valid.sum(dim=1).to(dtype=torch.int64).tolist()
    native_counts = native_valid.sum(dim=1).to(dtype=torch.int64).tolist()
    sched_cache: dict[tuple[int, int], tuple[list[int], list[int], list[int]]] = {}
    schedules: list[tuple[list[int], list[int], list[int]]] = []
    max_k = 0

    for pred_len, native_len in zip(pred_counts, native_counts):
        key = (pred_len, native_len)
        sched = sched_cache.get(key)
        if sched is None:
            pred_starts: list[int] = []
            native_starts: list[int] = []
            overlap_lens: list[int] = []
            if min(pred_len, native_len) >= 3:
                for offset in range(-(native_len - 3), pred_len - 2):
                    pred_start = max(offset, 0)
                    native_start = max(-offset, 0)
                    overlap = min(pred_len - pred_start, native_len - native_start)
                    if overlap < 3:
                        continue
                    pred_starts.append(pred_start)
                    native_starts.append(native_start)
                    overlap_lens.append(overlap)
            sched = (pred_starts, native_starts, overlap_lens)
            sched_cache[key] = sched
        schedules.append(sched)
        max_k = max(max_k, len(sched[0]))

    pred_starts_all = torch.zeros((B, max_k), dtype=torch.long, device=device)
    native_starts_all = torch.zeros((B, max_k), dtype=torch.long, device=device)
    overlap_lens_all = torch.zeros((B, max_k), dtype=torch.long, device=device)
    cand_mask = torch.zeros((B, max_k), dtype=torch.bool, device=device)

    for b, (pred_starts, native_starts, overlap_lens) in enumerate(schedules):
        if not overlap_lens:
            continue
        count = len(overlap_lens)
        pred_starts_all[b, :count] = torch.tensor(pred_starts, dtype=torch.long, device=device)
        native_starts_all[b, :count] = torch.tensor(native_starts, dtype=torch.long, device=device)
        overlap_lens_all[b, :count] = torch.tensor(overlap_lens, dtype=torch.long, device=device)
        cand_mask[b, :count] = True

    return pred_starts_all, native_starts_all, overlap_lens_all, cand_mask


def _evaluate_local_fragment_dp_seeds(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    pred_valid: torch.Tensor,
    native_valid: torch.Tensor,
    d0: torch.Tensor,
    d0_search: torch.Tensor,
    Lnorm: torch.Tensor,
    max_mem_gb: float,
    dense: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate cross-index local fragment superpositions followed by one NW step."""
    B, N, _ = pred.shape
    device = pred.device
    dtype = pred.dtype
    eye3 = torch.eye(3, dtype=dtype, device=device)
    best_R = eye3.unsqueeze(0).expand(B, 3, 3).clone()
    best_t = torch.zeros(B, 3, dtype=dtype, device=device)
    if B == 0:
        return best_R, best_t, torch.zeros((0,), dtype=dtype, device=device)

    pred_starts, native_starts, frag_lens, cand_mask = _build_pairwise_local_fragment_starts(
        pred_valid,
        native_valid,
        dense=dense,
    )
    active = torch.nonzero(cand_mask, as_tuple=False)
    if active.numel() == 0:
        return best_R, best_t, torch.full((B,), -torch.inf, dtype=dtype, device=device)

    itemsize = torch.empty((), dtype=dtype).element_size()
    bytes_per_cand = max(1, N * N * itemsize * 6)
    budget = max(1, int(max_mem_gb * (1024 ** 3) * 0.15))
    chunk = max(1, budget // bytes_per_cand)
    chunk = min(chunk, active.shape[0])
    best_score_cpu = [float("-inf")] * B

    for start in range(0, active.shape[0], chunk):
        end = min(start + chunk, active.shape[0])
        pair_idx = active[start:end, 0]
        cand_idx = active[start:end, 1]
        count = pair_idx.shape[0]

        pred_sel = pred.index_select(0, pair_idx)
        native_sel = native.index_select(0, pair_idx)
        valid_sel = valid.index_select(0, pair_idx)
        pred_valid_sel = pred_valid.index_select(0, pair_idx)
        native_valid_sel = native_valid.index_select(0, pair_idx)
        d0_sel = d0.index_select(0, pair_idx)
        lnorm_sel = Lnorm.index_select(0, pair_idx)
        frag_len_sel = frag_lens[pair_idx, cand_idx]

        max_frag = int(frag_len_sel.max().item())
        offsets = torch.arange(max_frag, device=device).unsqueeze(0)
        pred_idx = pred_starts[pair_idx, cand_idx].unsqueeze(1) + offsets
        native_idx = native_starts[pair_idx, cand_idx].unsqueeze(1) + offsets
        frag_mask = offsets < frag_len_sel.unsqueeze(1)
        pred_idx = torch.where(frag_mask, pred_idx, torch.zeros_like(pred_idx))
        native_idx = torch.where(frag_mask, native_idx, torch.zeros_like(native_idx))
        local_idx = torch.arange(count, device=device).unsqueeze(1)
        pred_frag = pred_sel[local_idx, pred_idx]
        native_frag = native_sel[local_idx, native_idx]

        seed_R, seed_t = rt.kabsch_batch(pred_frag, native_frag, frag_mask)
        dp_R, dp_t, dp_score = _dp._dp_refine(
            pred_sel,
            native_sel,
            valid_sel,
            seed_R,
            seed_t,
            d0_sel,
            d0_search.index_select(0, pair_idx),
            None,
            lnorm_sel,
            max_iter=2,
            max_mem_gb=max_mem_gb,
            pred_valid=pred_valid_sel,
            native_valid=native_valid_sel,
        )

        pair_ids = pair_idx.tolist()
        scores = dp_score.tolist()
        best_pos: dict[int, int] = {}
        best_score_chunk: dict[int, float] = {}
        for pos, (pair_id, score) in enumerate(zip(pair_ids, scores)):
            prev = best_score_chunk.get(pair_id)
            if prev is None or score > prev:
                best_score_chunk[pair_id] = score
                best_pos[pair_id] = pos
        for pair_id, pos in best_pos.items():
            score = best_score_chunk[pair_id]
            if score <= best_score_cpu[pair_id]:
                continue
            best_score_cpu[pair_id] = score
            best_R[pair_id] = dp_R[pos]
            best_t[pair_id] = dp_t[pos]

    best_score = torch.tensor(best_score_cpu, dtype=dtype, device=device)
    return best_R, best_t, best_score


def _evaluate_threading_dp_seeds(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    pred_valid: torch.Tensor,
    native_valid: torch.Tensor,
    d0: torch.Tensor,
    d0_search: torch.Tensor,
    Lnorm: torch.Tensor,
    max_mem_gb: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate gapless threading seeds, then refine them with detailed search and DP."""
    B, N, _ = pred.shape
    device = pred.device
    dtype = pred.dtype
    eye3 = torch.eye(3, dtype=dtype, device=device)
    best_R = eye3.unsqueeze(0).expand(B, 3, 3).clone()
    best_t = torch.zeros(B, 3, dtype=dtype, device=device)
    if B == 0:
        return best_R, best_t, torch.zeros((0,), dtype=dtype, device=device)

    pred_starts, native_starts, overlap_lens, cand_mask = _build_pairwise_thread_starts(
        pred_valid,
        native_valid,
    )
    active = torch.nonzero(cand_mask, as_tuple=False)
    if active.numel() == 0:
        return best_R, best_t, torch.full((B,), -torch.inf, dtype=dtype, device=device)

    itemsize = torch.empty((), dtype=dtype).element_size()
    bytes_per_cand = max(1, N * N * itemsize * 8)
    budget = max(1, int(max_mem_gb * (1024 ** 3) * 0.10))
    chunk = max(1, budget // bytes_per_cand)
    chunk = min(chunk, active.shape[0])
    best_score_cpu = [float("-inf")] * B

    for start in range(0, active.shape[0], chunk):
        end = min(start + chunk, active.shape[0])
        pair_idx = active[start:end, 0]
        cand_idx = active[start:end, 1]

        pred_sel = pred.index_select(0, pair_idx)
        native_sel = native.index_select(0, pair_idx)
        valid_sel = valid.index_select(0, pair_idx)
        pred_valid_sel = pred_valid.index_select(0, pair_idx)
        native_valid_sel = native_valid.index_select(0, pair_idx)
        d0_sel = d0.index_select(0, pair_idx)
        d0_search_sel = d0_search.index_select(0, pair_idx)
        lnorm_sel = Lnorm.index_select(0, pair_idx)
        overlap_len_sel = overlap_lens[pair_idx, cand_idx]

        max_pairs = int(overlap_len_sel.max().item())
        offsets = torch.arange(max_pairs, device=device).unsqueeze(0)
        align_p = pred_starts[pair_idx, cand_idx].unsqueeze(1) + offsets
        align_n = native_starts[pair_idx, cand_idx].unsqueeze(1) + offsets
        pair_mask = offsets < overlap_len_sel.unsqueeze(1)
        align_p = torch.where(pair_mask, align_p, torch.zeros_like(align_p))
        align_n = torch.where(pair_mask, align_n, torch.zeros_like(align_n))

        seed_R, seed_t, seed_score = _dp._alignment_detailed_search(
            pred_sel,
            native_sel,
            align_p,
            align_n,
            pair_mask,
            d0_sel,
            d0_search_sel,
            None,
            lnorm_sel,
            max_mem_gb,
        )
        dp_R, dp_t, dp_score = _dp._dp_refine(
            pred_sel,
            native_sel,
            valid_sel,
            seed_R,
            seed_t,
            d0_sel,
            d0_search_sel,
            None,
            lnorm_sel,
            max_iter=1,
            max_mem_gb=max_mem_gb,
            pred_valid=pred_valid_sel,
            native_valid=native_valid_sel,
        )
        use_dp = dp_score > seed_score
        cand_R = torch.where(use_dp[:, None, None], dp_R, seed_R)
        cand_t = torch.where(use_dp[:, None], dp_t, seed_t)
        cand_score = torch.where(use_dp, dp_score, seed_score)

        pair_ids = pair_idx.tolist()
        scores = cand_score.tolist()
        best_pos: dict[int, int] = {}
        best_score_chunk: dict[int, float] = {}
        for pos, (pair_id, score) in enumerate(zip(pair_ids, scores)):
            prev = best_score_chunk.get(pair_id)
            if prev is None or score > prev:
                best_score_chunk[pair_id] = score
                best_pos[pair_id] = pos
        for pair_id, pos in best_pos.items():
            score = best_score_chunk[pair_id]
            if score <= best_score_cpu[pair_id]:
                continue
            best_score_cpu[pair_id] = score
            best_R[pair_id] = cand_R[pos]
            best_t[pair_id] = cand_t[pos]

    best_score = torch.tensor(best_score_cpu, dtype=dtype, device=device)
    return best_R, best_t, best_score
