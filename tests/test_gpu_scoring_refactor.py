from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scoring._local_validation_gpu.runtime as lvgr
import scoring._tmscore_gpu.search as tgs
import scoring._tmscore_gpu.runtime as tgr
import scoring.local_validation_gpu as lvg
import scoring.tmscore_gpu as tg


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _configure_strict_float64() -> None:
    tg.reset_backend_runtime_state()
    lvg._configure_tmscore_backend("strict")
    tg._DEFAULT_DTYPE = torch.float64
    lvg._DEFAULT_DTYPE = torch.float64


def test_compatibility_modules_share_runtime_state() -> None:
    assert tg is tgr
    assert lvg is lvgr

    old_tg_dtype = tg._DEFAULT_DTYPE
    old_lvg_dtype = lvg._DEFAULT_DTYPE
    try:
        tg._DEFAULT_DTYPE = torch.float64
        lvg._DEFAULT_DTYPE = torch.float16
        assert tgr._DEFAULT_DTYPE == torch.float64
        assert lvgr._DEFAULT_DTYPE == torch.float16
    finally:
        tg._DEFAULT_DTYPE = old_tg_dtype
        lvg._DEFAULT_DTYPE = old_lvg_dtype


def test_contact_seed_zero_offset_is_deduped() -> None:
    tg.reset_backend_runtime_state()
    tg.configure_pair_local_search(
        contact_seed_mults=(1.0, 1.5),
        contact_seed_add_offsets=(0.0, 2.0),
    )
    radii = tg._contact_seed_radii(torch.tensor([4.5], dtype=torch.float64))
    assert torch.allclose(radii, torch.tensor([[4.5, 6.75, 6.5]], dtype=torch.float64))


def test_anchor_contact_seed_masks_are_unique() -> None:
    tg.reset_backend_runtime_state()
    tg.configure_pair_local_search(
        contact_seed_mults=(10.0,),
        contact_seed_add_offsets=(),
        contact_seed_max_anchors=8,
    )
    device = _device()
    pred = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
        dtype=torch.float64,
        device=device,
    )
    valid = torch.ones((1, 4), dtype=torch.bool, device=device)
    seeds = tg._build_pairwise_anchor_contact_seed_masks(
        pred,
        pred.clone(),
        valid,
        torch.tensor([1.0], dtype=torch.float64, device=device),
    )
    assert seeds.shape == (1, 1, 4)
    assert torch.equal(seeds[0, 0].cpu(), torch.tensor([True, True, True, True]))


def test_selective_weighted_refine_topk_is_global_across_chunks() -> None:
    tg.reset_backend_runtime_state()
    tg.configure_pair_local_search(
        tm_weighted_refine_iters=1,
        tm_weighted_refine_topk=1,
        tm_weighted_refine_score_margin=0.0,
    )

    pred = torch.zeros((1, 4, 3), dtype=torch.float64)
    native = pred.clone()
    valid = torch.ones((1, 4), dtype=torch.bool)
    seed_bank = torch.ones((1, 1, 4), dtype=torch.bool)
    d0_candidates = torch.arange(1.0, 9.0, dtype=torch.float64).unsqueeze(0)
    d0 = torch.ones((1,), dtype=torch.float64)
    score_d8 = torch.full((1,), 100.0, dtype=torch.float64)
    Lnorm = torch.full((1,), 4.0, dtype=torch.float64)

    orig_plan = tgr._pairseed_chunk_plan
    orig_iter = tgs._iterative_seed_refine
    orig_score = tgr._tm_score_fused
    orig_refine = tgr._tm_weighted_refine
    refined = []

    def fake_plan(B: int, N: int, KM: int, dtype: torch.dtype, max_mem_gb: float) -> tuple[int, int]:
        return 1, 2

    def fake_iterative(
        pred: torch.Tensor,
        native: torch.Tensor,
        valid: torch.Tensor,
        seed_mask: torch.Tensor,
        d0_search: torch.Tensor,
        max_iter: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eye = torch.eye(3, dtype=pred.dtype, device=pred.device).unsqueeze(0).expand(pred.shape[0], 3, 3).clone()
        t = torch.zeros((pred.shape[0], 3), dtype=pred.dtype, device=pred.device)
        t[:, 0] = d0_search
        return eye, t

    def fake_score(
        pred: torch.Tensor,
        native: torch.Tensor,
        valid: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        d0: torch.Tensor,
        Lnorm: torch.Tensor,
        score_d8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return t[:, 0].clone()

    def fake_refine(
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
        refined.extend(float(v) for v in t[:, 0].cpu().tolist())
        return R, t, t[:, 0].clone()

    try:
        tgr._pairseed_chunk_plan = fake_plan
        tgs._iterative_seed_refine = fake_iterative
        tgr._tm_score_fused = fake_score
        tgr._tm_weighted_refine = fake_refine

        _, _, best_score = tgs._evaluate_seed_bank(
            pred,
            native,
            valid,
            seed_bank,
            d0_candidates,
            d0,
            score_d8,
            Lnorm,
            max_iter=1,
            max_mem_gb=1.0,
        )
    finally:
        tgr._pairseed_chunk_plan = orig_plan
        tgs._iterative_seed_refine = orig_iter
        tgr._tm_score_fused = orig_score
        tgr._tm_weighted_refine = orig_refine
        tg.reset_backend_runtime_state()

    assert refined == [8.0]
    assert torch.allclose(best_score, torch.tensor([8.0], dtype=torch.float64))


def test_tmscore_search_snapshot() -> None:
    _configure_strict_float64()
    device = _device()

    pred = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.2, 0.3]]],
        dtype=torch.float64,
        device=device,
    )
    native = torch.tensor(
        [[[2.0, 3.0, 4.0], [2.8, 3.4, 4.1], [1.7, 4.1, 4.2], [2.5, 4.5, 4.3], [2.1, 3.6, 4.6]]],
        dtype=torch.float64,
        device=device,
    )
    valid = torch.ones((1, 5), dtype=torch.bool, device=device)
    Lnorm = valid.sum(dim=1).to(dtype=torch.float64)
    d0, d0_search, score_d8 = tg.d0_from_length(Lnorm, dtype=torch.float64)

    score = tg.tmscore_search(
        pred,
        native,
        valid,
        d0,
        d0_search,
        score_d8,
        Lnorm,
        max_iter=20,
        use_fragment_search=True,
        max_mem_gb=4.0,
        dp_iter=0,
    )

    assert torch.allclose(score.cpu(), torch.tensor([0.7666330261384652], dtype=torch.float64), atol=1e-12, rtol=0.0)


def test_score_target_handles_multimer_permutation() -> None:
    _configure_strict_float64()

    native = pd.DataFrame(
        [
            {"ID": "toy_1", "resname": "A", "resid": 1, "chain": "A", "copy": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0},
            {"ID": "toy_2", "resname": "A", "resid": 2, "chain": "A", "copy": 1, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0},
            {"ID": "toy_3", "resname": "A", "resid": 3, "chain": "A", "copy": 2, "x_1": 5.0, "y_1": 0.0, "z_1": 0.0},
            {"ID": "toy_4", "resname": "A", "resid": 4, "chain": "A", "copy": 2, "x_1": 6.0, "y_1": 0.0, "z_1": 0.0},
        ]
    )
    predicted = pd.DataFrame(
        [
            {"ID": "toy_1", "resname": "A", "resid": 1, "x_1": 5.0, "y_1": 0.0, "z_1": 0.0},
            {"ID": "toy_2", "resname": "A", "resid": 2, "x_1": 6.0, "y_1": 0.0, "z_1": 0.0},
            {"ID": "toy_3", "resname": "A", "resid": 3, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0},
            {"ID": "toy_4", "resname": "A", "resid": 4, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0},
        ]
    )

    score = lvg.score_target(
        target_id="toy",
        group_native=native,
        group_predicted=predicted,
        device=_device(),
        max_iter=20,
        use_fragment_search=True,
        max_mem_gb=4.0,
        dp_iter=0,
    )

    assert score == 1.0
