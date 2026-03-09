"""Coordinate extraction utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch

from . import runtime as rt


def _sorted_by_resid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__resid_int"] = out["resid"].astype(int)
    out = out.sort_values("__resid_int").reset_index(drop=True)
    return out


def _available_frames(df: pd.DataFrame, max_frames: int) -> list[int]:
    ids: list[int] = []
    cols = set(df.columns.tolist())
    for i in range(1, max_frames + 1):
        if f"x_{i}" in cols and f"y_{i}" in cols and f"z_{i}" in cols:
            ids.append(i)
    return ids


def extract_coords(
    df: pd.DataFrame, frame_ids: list[int], device: str,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert frame coordinate columns into tensors.

    Returns
    -------
    coords:
        (F, L, 3)
    valid:
        (F, L) bool
    """
    if dtype is None:
        dtype = rt._DEFAULT_DTYPE

    if not frame_ids:
        empty_c = torch.zeros((0, len(df), 3), dtype=dtype, device=device)
        empty_v = torch.zeros((0, len(df)), dtype=torch.bool, device=device)
        return empty_c, empty_v

    arrs: list[np.ndarray] = []
    for fid in frame_ids:
        cols = [f"x_{fid}", f"y_{fid}", f"z_{fid}"]
        arr = df.loc[:, cols].to_numpy(dtype=np.float64, copy=False)
        arrs.append(arr)
    stacked = np.stack(arrs, axis=0)  # (F, L, 3)
    valid_np = np.all(stacked > rt.INVALID_COORD, axis=2)

    coords = torch.from_numpy(stacked).to(device=device, dtype=dtype)
    coords = torch.nan_to_num(coords, nan=0.0)
    valid = torch.from_numpy(valid_np).to(device=device, dtype=torch.bool)
    return coords, valid
