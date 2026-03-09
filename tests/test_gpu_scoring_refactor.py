from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scoring._local_validation_gpu.runtime as lvgr
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
