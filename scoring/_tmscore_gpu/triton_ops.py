"""Triton kernel definitions and launch configuration for GPU TM-score."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

import scoring._tmscore_gpu.runtime as rt


def _triton_select_launch_config(device: torch.device, N: int) -> tuple[int, int, int]:
    """Launch config for _select_mask_kernel.

    Blackwell (SM120+) benefits from a slightly more aggressive launch config
    on this memory-heavy kernel. Older architectures retain the previous
    conservative defaults.
    """
    if rt._is_blackwell(device):
        # SM120 autotune anchors (RTX 5090):
        #   N=64  -> (128,4,3)
        #   N=128 -> (128,2,3)
        #   N=320 -> (512,8,3)
        #   N=512 -> (512,8,4)
        #   N=640 -> (256,4,3)
        if N >= 640:
            block_n, num_warps, num_stages = 256, 4, 3
        elif N >= 448:
            block_n, num_warps, num_stages = 512, 8, 4
        elif N >= 192:
            block_n, num_warps, num_stages = 512, 8, 3
        elif N >= 96:
            block_n, num_warps, num_stages = 128, 2, 3
        elif N >= 64:
            block_n, num_warps, num_stages = 128, 4, 3
        else:
            block_n, num_warps, num_stages = 64, 2, 2
    else:
        block_n = 256 if N >= 256 else (128 if N >= 128 else 64)
        num_warps, num_stages = 4, 2

    return (
        rt._env_int("BAMBOO_TRITON_SELECT_BLOCK_N", block_n) or block_n,
        rt._env_int("BAMBOO_TRITON_SELECT_NUM_WARPS", num_warps) or num_warps,
        rt._env_int("BAMBOO_TRITON_SELECT_NUM_STAGES", num_stages) or num_stages,
    )


def _triton_kabsch_launch_config(device: torch.device, N: int) -> tuple[int, int, int]:
    """Launch config for _kabsch_fused_kernel.

    The fused Kabsch kernel is register-heavy; Blackwell generally tolerates
    higher warp counts at moderate BLOCK_N, but too many stages can hurt
    occupancy. Tune for throughput on RTX 5090 while keeping conservative
    settings on older GPUs.
    """
    if rt._is_blackwell(device):
        # SM120 autotune anchors (RTX 5090):
        #   N=96  -> (128,2,2)
        #   N=256 -> (256,4,4)
        #   N=384 -> (512,4,4)
        #   N=512 -> (512,4,4)
        if N >= 384:
            block_n, num_warps, num_stages = 512, 4, 4
        elif N >= 192:
            block_n, num_warps, num_stages = 256, 4, 4
        elif N >= 64:
            block_n, num_warps, num_stages = 128, 2, 2
        else:
            block_n, num_warps, num_stages = 64, 2, 2
    else:
        block_n = 256 if N >= 256 else (128 if N >= 128 else 64)
        num_warps, num_stages = 4, 2

    return (
        rt._env_int("BAMBOO_TRITON_KABSCH_BLOCK_N", block_n) or block_n,
        rt._env_int("BAMBOO_TRITON_KABSCH_NUM_WARPS", num_warps) or num_warps,
        rt._env_int("BAMBOO_TRITON_KABSCH_NUM_STAGES", num_stages) or num_stages,
    )


def _can_use_triton_kabsch(P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor) -> bool:
    return bool(
        _HAS_TRITON
        and rt._ENABLE_TRITON_KABSCH
        and P.is_cuda
        and Q.is_cuda
        and mask.is_cuda
        and P.dtype == torch.float32
        and Q.dtype == torch.float32
        and P.is_contiguous()
        and Q.is_contiguous()
        and mask.is_contiguous()
        and P.shape[0] >= rt._TRITON_MIN_B
        and P.shape[1] >= rt._TRITON_MIN_N
    )


def _can_use_triton_refine(pred: torch.Tensor, native: torch.Tensor, valid: torch.Tensor) -> bool:
    return bool(
        _HAS_TRITON
        and rt._ENABLE_TRITON_REFINE
        and pred.is_cuda
        and native.is_cuda
        and valid.is_cuda
        and pred.dtype == torch.float32
        and native.dtype == torch.float32
        and pred.is_contiguous()
        and native.is_contiguous()
        and valid.is_contiguous()
        and pred.shape[0] >= rt._TRITON_MIN_B
        and pred.shape[1] >= rt._TRITON_MIN_N
    )


if _HAS_TRITON:
    @triton.jit
    def _select_mask_kernel(
        P_ptr, Q_ptr, R_ptr, T_ptr, V_ptr, D0SQ_ptr, OUT_ptr, FLAG_ptr,
        B, N,
        stride_pb, stride_pn, stride_pc,
        stride_qb, stride_qn, stride_qc,
        stride_rb, stride_r0, stride_r1,
        stride_tb, stride_tn,
        stride_vb, stride_vn,
        stride_ob, stride_on,
        stride_fb,
        ABS_TOL, REL_TOL,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        pid_n = tl.program_id(1)
        if b >= B:
            return

        offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        in_bounds = offs < N

        p0 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 0 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32)
        p1 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 1 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32)
        p2 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 2 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32)
        q0 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 0 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32)
        q1 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 1 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32)
        q2 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 2 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + b * stride_vb + offs * stride_vn, mask=in_bounds, other=0).to(tl.int32)

        r00 = tl.load(R_ptr + b * stride_rb + 0 * stride_r0 + 0 * stride_r1).to(tl.float32)
        r01 = tl.load(R_ptr + b * stride_rb + 0 * stride_r0 + 1 * stride_r1).to(tl.float32)
        r02 = tl.load(R_ptr + b * stride_rb + 0 * stride_r0 + 2 * stride_r1).to(tl.float32)
        r10 = tl.load(R_ptr + b * stride_rb + 1 * stride_r0 + 0 * stride_r1).to(tl.float32)
        r11 = tl.load(R_ptr + b * stride_rb + 1 * stride_r0 + 1 * stride_r1).to(tl.float32)
        r12 = tl.load(R_ptr + b * stride_rb + 1 * stride_r0 + 2 * stride_r1).to(tl.float32)
        r20 = tl.load(R_ptr + b * stride_rb + 2 * stride_r0 + 0 * stride_r1).to(tl.float32)
        r21 = tl.load(R_ptr + b * stride_rb + 2 * stride_r0 + 1 * stride_r1).to(tl.float32)
        r22 = tl.load(R_ptr + b * stride_rb + 2 * stride_r0 + 2 * stride_r1).to(tl.float32)
        t0 = tl.load(T_ptr + b * stride_tb + 0 * stride_tn).to(tl.float32)
        t1 = tl.load(T_ptr + b * stride_tb + 1 * stride_tn).to(tl.float32)
        t2 = tl.load(T_ptr + b * stride_tb + 2 * stride_tn).to(tl.float32)
        d0sq = tl.load(D0SQ_ptr + b).to(tl.float32)

        m0 = p0 * r00 + p1 * r10 + p2 * r20 + t0
        m1 = p0 * r01 + p1 * r11 + p2 * r21 + t1
        m2 = p0 * r02 + p1 * r12 + p2 * r22 + t2

        d0 = m0 - q0
        d1 = m1 - q1
        d2 = m2 - q2
        dist2 = d0 * d0 + d1 * d1 + d2 * d2
        valid = v != 0
        sel = valid & (dist2 <= d0sq)
        tl.store(OUT_ptr + b * stride_ob + offs * stride_on, sel.to(tl.uint8), mask=in_bounds)
        tol = ABS_TOL + REL_TOL * d0sq
        near = valid & (tl.abs(dist2 - d0sq) <= tol)
        block_has_near = tl.max(near.to(tl.int32), axis=0)
        tl.atomic_or(FLAG_ptr + b * stride_fb, block_has_near)

    @triton.jit
    def _kabsch_fused_kernel(
        P_ptr, Q_ptr, M_ptr, R_ptr, T_ptr,
        B, N,
        stride_pb, stride_pn, stride_pc,
        stride_qb, stride_qn, stride_qc,
        stride_mb, stride_mn,
        stride_rb, stride_r0, stride_r1,
        stride_tb, stride_tn,
        BLOCK_N: tl.constexpr,
    ):
        """Single-launch fused Kabsch: (P, Q, mask) → (R, t).

        One program per batch element. Two passes over N:
          Pass 1: masked centroid (count, sumP, sumQ)
          Pass 2: centered covariance (9 accumulators)
        Then inline Jacobi eigensolve → quat → rotation + translation.
        """
        b = tl.program_id(0)
        if b >= B:
            return

        offs_base = tl.arange(0, BLOCK_N)

        # --- Pass 1: centroid ---
        acc_count = tl.zeros((), dtype=tl.float32)
        spx = tl.zeros((), dtype=tl.float32)
        spy = tl.zeros((), dtype=tl.float32)
        spz = tl.zeros((), dtype=tl.float32)
        sqx = tl.zeros((), dtype=tl.float32)
        sqy = tl.zeros((), dtype=tl.float32)
        sqz = tl.zeros((), dtype=tl.float32)

        for n0 in tl.range(0, N, BLOCK_N):
            offs = n0 + offs_base
            in_bounds = offs < N
            m = tl.load(M_ptr + b * stride_mb + offs * stride_mn, mask=in_bounds, other=0).to(tl.float32)
            p0 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 0 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32)
            p1 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 1 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32)
            p2 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 2 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32)
            q0 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 0 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32)
            q1 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 1 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32)
            q2 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 2 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32)
            acc_count += tl.sum(m, axis=0)
            spx += tl.sum(m * p0, axis=0)
            spy += tl.sum(m * p1, axis=0)
            spz += tl.sum(m * p2, axis=0)
            sqx += tl.sum(m * q0, axis=0)
            sqy += tl.sum(m * q1, axis=0)
            sqz += tl.sum(m * q2, axis=0)

        safe_count = tl.maximum(acc_count, 1.0)
        cp0 = spx / safe_count; cp1 = spy / safe_count; cp2 = spz / safe_count
        cq0 = sqx / safe_count; cq1 = sqy / safe_count; cq2 = sqz / safe_count

        # --- Pass 2: centered covariance ---
        c00 = tl.zeros((), dtype=tl.float32); c01 = tl.zeros((), dtype=tl.float32); c02 = tl.zeros((), dtype=tl.float32)
        c10 = tl.zeros((), dtype=tl.float32); c11 = tl.zeros((), dtype=tl.float32); c12 = tl.zeros((), dtype=tl.float32)
        c20 = tl.zeros((), dtype=tl.float32); c21 = tl.zeros((), dtype=tl.float32); c22 = tl.zeros((), dtype=tl.float32)

        for n0 in tl.range(0, N, BLOCK_N):
            offs = n0 + offs_base
            in_bounds = offs < N
            m = tl.load(M_ptr + b * stride_mb + offs * stride_mn, mask=in_bounds, other=0).to(tl.float32)
            p0 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 0 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32) - cp0
            p1 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 1 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32) - cp1
            p2 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 2 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32) - cp2
            q0 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 0 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32) - cq0
            q1 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 1 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32) - cq1
            q2 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 2 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32) - cq2
            wp0 = m * p0; wp1 = m * p1; wp2 = m * p2
            c00 += tl.sum(wp0 * q0, axis=0); c01 += tl.sum(wp0 * q1, axis=0); c02 += tl.sum(wp0 * q2, axis=0)
            c10 += tl.sum(wp1 * q0, axis=0); c11 += tl.sum(wp1 * q1, axis=0); c12 += tl.sum(wp1 * q2, axis=0)
            c20 += tl.sum(wp2 * q0, axis=0); c21 += tl.sum(wp2 * q1, axis=0); c22 += tl.sum(wp2 * q2, axis=0)

        # --- Build Horn matrix F ---
        f00 = c00 + c11 + c22
        f01 = c12 - c21
        f02 = c20 - c02
        f03 = c01 - c10
        f11 = c00 - c11 - c22
        f12 = c01 + c10
        f13 = c20 + c02
        f22 = -c00 + c11 - c22
        f23 = c12 + c21
        f33 = -c00 - c11 + c22

        a00 = f00; a01 = f01; a02 = f02; a03 = f03
        a10 = f01; a11 = f11; a12 = f12; a13 = f13
        a20 = f02; a21 = f12; a22 = f22; a23 = f23
        a30 = f03; a31 = f13; a32 = f23; a33 = f33

        # V = I (scalars for single batch element)
        v00 = 1.0; v01 = 0.0; v02 = 0.0; v03 = 0.0
        v10 = 0.0; v11 = 1.0; v12 = 0.0; v13 = 0.0
        v20 = 0.0; v21 = 0.0; v22 = 1.0; v23 = 0.0
        v30 = 0.0; v31 = 0.0; v32 = 0.0; v33 = 1.0

        # --- 6 sweeps x 6 Givens rotations ---
        # 4 sweeps suffice for most inputs, but near-degenerate eigenspectra
        # (eigenvalues within ~1e-3 of each other) can require 5-6 sweeps
        # for float32 convergence.  Cost is negligible (scalar register ops).
        EPS: tl.constexpr = 1e-20
        for _sweep in range(6):
            # pair (0,1)
            apq = a01; nz = tl.abs(apq) > EPS
            dif = a11 - a00; tau = dif / (2.0 * apq + (~nz) * EPS)
            t_ = tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            t_ = tl.where(nz, t_, 0.0); c = 1.0 / tl.sqrt(1.0 + t_ * t_); s = t_ * c; c = tl.where(nz, c, 1.0)
            tmp0=c*a00-s*a01; tmp1=s*a00+c*a01; a00=tmp0; a01=tmp1
            tmp0=c*a10-s*a11; tmp1=s*a10+c*a11; a10=tmp0; a11=tmp1
            tmp0=c*a20-s*a21; tmp1=s*a20+c*a21; a20=tmp0; a21=tmp1
            tmp0=c*a30-s*a31; tmp1=s*a30+c*a31; a30=tmp0; a31=tmp1
            tmp0=c*a00-s*a10; tmp1=s*a00+c*a10; a00=tmp0; a10=tmp1
            tmp0=c*a01-s*a11; tmp1=s*a01+c*a11; a01=tmp0; a11=tmp1
            tmp0=c*a02-s*a12; tmp1=s*a02+c*a12; a02=tmp0; a12=tmp1
            tmp0=c*a03-s*a13; tmp1=s*a03+c*a13; a03=tmp0; a13=tmp1
            tmp0=c*v00-s*v01; tmp1=s*v00+c*v01; v00=tmp0; v01=tmp1
            tmp0=c*v10-s*v11; tmp1=s*v10+c*v11; v10=tmp0; v11=tmp1
            tmp0=c*v20-s*v21; tmp1=s*v20+c*v21; v20=tmp0; v21=tmp1
            tmp0=c*v30-s*v31; tmp1=s*v30+c*v31; v30=tmp0; v31=tmp1
            # pair (0,2)
            apq = a02; nz = tl.abs(apq) > EPS
            dif = a22 - a00; tau = dif / (2.0 * apq + (~nz) * EPS)
            t_ = tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            t_ = tl.where(nz, t_, 0.0); c = 1.0 / tl.sqrt(1.0 + t_ * t_); s = t_ * c; c = tl.where(nz, c, 1.0)
            tmp0=c*a00-s*a02; tmp1=s*a00+c*a02; a00=tmp0; a02=tmp1
            tmp0=c*a10-s*a12; tmp1=s*a10+c*a12; a10=tmp0; a12=tmp1
            tmp0=c*a20-s*a22; tmp1=s*a20+c*a22; a20=tmp0; a22=tmp1
            tmp0=c*a30-s*a32; tmp1=s*a30+c*a32; a30=tmp0; a32=tmp1
            tmp0=c*a00-s*a20; tmp1=s*a00+c*a20; a00=tmp0; a20=tmp1
            tmp0=c*a01-s*a21; tmp1=s*a01+c*a21; a01=tmp0; a21=tmp1
            tmp0=c*a02-s*a22; tmp1=s*a02+c*a22; a02=tmp0; a22=tmp1
            tmp0=c*a03-s*a23; tmp1=s*a03+c*a23; a03=tmp0; a23=tmp1
            tmp0=c*v00-s*v02; tmp1=s*v00+c*v02; v00=tmp0; v02=tmp1
            tmp0=c*v10-s*v12; tmp1=s*v10+c*v12; v10=tmp0; v12=tmp1
            tmp0=c*v20-s*v22; tmp1=s*v20+c*v22; v20=tmp0; v22=tmp1
            tmp0=c*v30-s*v32; tmp1=s*v30+c*v32; v30=tmp0; v32=tmp1
            # pair (0,3)
            apq = a03; nz = tl.abs(apq) > EPS
            dif = a33 - a00; tau = dif / (2.0 * apq + (~nz) * EPS)
            t_ = tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            t_ = tl.where(nz, t_, 0.0); c = 1.0 / tl.sqrt(1.0 + t_ * t_); s = t_ * c; c = tl.where(nz, c, 1.0)
            tmp0=c*a00-s*a03; tmp1=s*a00+c*a03; a00=tmp0; a03=tmp1
            tmp0=c*a10-s*a13; tmp1=s*a10+c*a13; a10=tmp0; a13=tmp1
            tmp0=c*a20-s*a23; tmp1=s*a20+c*a23; a20=tmp0; a23=tmp1
            tmp0=c*a30-s*a33; tmp1=s*a30+c*a33; a30=tmp0; a33=tmp1
            tmp0=c*a00-s*a30; tmp1=s*a00+c*a30; a00=tmp0; a30=tmp1
            tmp0=c*a01-s*a31; tmp1=s*a01+c*a31; a01=tmp0; a31=tmp1
            tmp0=c*a02-s*a32; tmp1=s*a02+c*a32; a02=tmp0; a32=tmp1
            tmp0=c*a03-s*a33; tmp1=s*a03+c*a33; a03=tmp0; a33=tmp1
            tmp0=c*v00-s*v03; tmp1=s*v00+c*v03; v00=tmp0; v03=tmp1
            tmp0=c*v10-s*v13; tmp1=s*v10+c*v13; v10=tmp0; v13=tmp1
            tmp0=c*v20-s*v23; tmp1=s*v20+c*v23; v20=tmp0; v23=tmp1
            tmp0=c*v30-s*v33; tmp1=s*v30+c*v33; v30=tmp0; v33=tmp1
            # pair (1,2)
            apq = a12; nz = tl.abs(apq) > EPS
            dif = a22 - a11; tau = dif / (2.0 * apq + (~nz) * EPS)
            t_ = tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            t_ = tl.where(nz, t_, 0.0); c = 1.0 / tl.sqrt(1.0 + t_ * t_); s = t_ * c; c = tl.where(nz, c, 1.0)
            tmp0=c*a01-s*a02; tmp1=s*a01+c*a02; a01=tmp0; a02=tmp1
            tmp0=c*a11-s*a12; tmp1=s*a11+c*a12; a11=tmp0; a12=tmp1
            tmp0=c*a21-s*a22; tmp1=s*a21+c*a22; a21=tmp0; a22=tmp1
            tmp0=c*a31-s*a32; tmp1=s*a31+c*a32; a31=tmp0; a32=tmp1
            tmp0=c*a11-s*a21; tmp1=s*a11+c*a21; a11=tmp0; a21=tmp1
            tmp0=c*a10-s*a20; tmp1=s*a10+c*a20; a10=tmp0; a20=tmp1
            tmp0=c*a12-s*a22; tmp1=s*a12+c*a22; a12=tmp0; a22=tmp1
            tmp0=c*a13-s*a23; tmp1=s*a13+c*a23; a13=tmp0; a23=tmp1
            tmp0=c*v01-s*v02; tmp1=s*v01+c*v02; v01=tmp0; v02=tmp1
            tmp0=c*v11-s*v12; tmp1=s*v11+c*v12; v11=tmp0; v12=tmp1
            tmp0=c*v21-s*v22; tmp1=s*v21+c*v22; v21=tmp0; v22=tmp1
            tmp0=c*v31-s*v32; tmp1=s*v31+c*v32; v31=tmp0; v32=tmp1
            # pair (1,3)
            apq = a13; nz = tl.abs(apq) > EPS
            dif = a33 - a11; tau = dif / (2.0 * apq + (~nz) * EPS)
            t_ = tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            t_ = tl.where(nz, t_, 0.0); c = 1.0 / tl.sqrt(1.0 + t_ * t_); s = t_ * c; c = tl.where(nz, c, 1.0)
            tmp0=c*a01-s*a03; tmp1=s*a01+c*a03; a01=tmp0; a03=tmp1
            tmp0=c*a11-s*a13; tmp1=s*a11+c*a13; a11=tmp0; a13=tmp1
            tmp0=c*a21-s*a23; tmp1=s*a21+c*a23; a21=tmp0; a23=tmp1
            tmp0=c*a31-s*a33; tmp1=s*a31+c*a33; a31=tmp0; a33=tmp1
            tmp0=c*a11-s*a31; tmp1=s*a11+c*a31; a11=tmp0; a31=tmp1
            tmp0=c*a10-s*a30; tmp1=s*a10+c*a30; a10=tmp0; a30=tmp1
            tmp0=c*a12-s*a32; tmp1=s*a12+c*a32; a12=tmp0; a32=tmp1
            tmp0=c*a13-s*a33; tmp1=s*a13+c*a33; a13=tmp0; a33=tmp1
            tmp0=c*v01-s*v03; tmp1=s*v01+c*v03; v01=tmp0; v03=tmp1
            tmp0=c*v11-s*v13; tmp1=s*v11+c*v13; v11=tmp0; v13=tmp1
            tmp0=c*v21-s*v23; tmp1=s*v21+c*v23; v21=tmp0; v23=tmp1
            tmp0=c*v31-s*v33; tmp1=s*v31+c*v33; v31=tmp0; v33=tmp1
            # pair (2,3)
            apq = a23; nz = tl.abs(apq) > EPS
            dif = a33 - a22; tau = dif / (2.0 * apq + (~nz) * EPS)
            t_ = tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            t_ = tl.where(nz, t_, 0.0); c = 1.0 / tl.sqrt(1.0 + t_ * t_); s = t_ * c; c = tl.where(nz, c, 1.0)
            tmp0=c*a02-s*a03; tmp1=s*a02+c*a03; a02=tmp0; a03=tmp1
            tmp0=c*a12-s*a13; tmp1=s*a12+c*a13; a12=tmp0; a13=tmp1
            tmp0=c*a22-s*a23; tmp1=s*a22+c*a23; a22=tmp0; a23=tmp1
            tmp0=c*a32-s*a33; tmp1=s*a32+c*a33; a32=tmp0; a33=tmp1
            tmp0=c*a22-s*a32; tmp1=s*a22+c*a32; a22=tmp0; a32=tmp1
            tmp0=c*a20-s*a30; tmp1=s*a20+c*a30; a20=tmp0; a30=tmp1
            tmp0=c*a21-s*a31; tmp1=s*a21+c*a31; a21=tmp0; a31=tmp1
            tmp0=c*a23-s*a33; tmp1=s*a23+c*a33; a23=tmp0; a33=tmp1
            tmp0=c*v02-s*v03; tmp1=s*v02+c*v03; v02=tmp0; v03=tmp1
            tmp0=c*v12-s*v13; tmp1=s*v12+c*v13; v12=tmp0; v13=tmp1
            tmp0=c*v22-s*v23; tmp1=s*v22+c*v23; v22=tmp0; v23=tmp1
            tmp0=c*v32-s*v33; tmp1=s*v32+c*v33; v32=tmp0; v33=tmp1

        # Max eigenvalue → eigenvector
        best_is_0 = (a00 >= a11) & (a00 >= a22) & (a00 >= a33)
        best_is_1 = (~best_is_0) & (a11 >= a22) & (a11 >= a33)
        best_is_2 = (~best_is_0) & (~best_is_1) & (a22 >= a33)
        q0 = tl.where(best_is_0, v00, tl.where(best_is_1, v01, tl.where(best_is_2, v02, v03)))
        q1 = tl.where(best_is_0, v10, tl.where(best_is_1, v11, tl.where(best_is_2, v12, v13)))
        q2 = tl.where(best_is_0, v20, tl.where(best_is_1, v21, tl.where(best_is_2, v22, v23)))
        q3 = tl.where(best_is_0, v30, tl.where(best_is_1, v31, tl.where(best_is_2, v32, v33)))

        qnorm = tl.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        qnorm = tl.maximum(qnorm, 1e-12)
        q0 = q0/qnorm; q1 = q1/qnorm; q2 = q2/qnorm; q3 = q3/qnorm

        r00 = q0*q0+q1*q1-q2*q2-q3*q3; r01 = 2.0*(q1*q2+q0*q3); r02 = 2.0*(q1*q3-q0*q2)
        r10 = 2.0*(q1*q2-q0*q3); r11 = q0*q0-q1*q1+q2*q2-q3*q3; r12 = 2.0*(q2*q3+q0*q1)
        r20 = 2.0*(q1*q3+q0*q2); r21 = 2.0*(q2*q3-q0*q1); r22 = q0*q0-q1*q1-q2*q2+q3*q3

        t0 = cq0 - (cp0*r00 + cp1*r10 + cp2*r20)
        t1 = cq1 - (cp0*r01 + cp1*r11 + cp2*r21)
        t2 = cq2 - (cp0*r02 + cp1*r12 + cp2*r22)

        degen = acc_count < 3.0
        r00 = tl.where(degen, 1.0, r00); r01 = tl.where(degen, 0.0, r01); r02 = tl.where(degen, 0.0, r02)
        r10 = tl.where(degen, 0.0, r10); r11 = tl.where(degen, 1.0, r11); r12 = tl.where(degen, 0.0, r12)
        r20 = tl.where(degen, 0.0, r20); r21 = tl.where(degen, 0.0, r21); r22 = tl.where(degen, 1.0, r22)
        t0 = tl.where(degen, 0.0, t0); t1 = tl.where(degen, 0.0, t1); t2 = tl.where(degen, 0.0, t2)

        rb = b * stride_rb
        tl.store(R_ptr + rb + 0*stride_r0 + 0*stride_r1, r00)
        tl.store(R_ptr + rb + 0*stride_r0 + 1*stride_r1, r01)
        tl.store(R_ptr + rb + 0*stride_r0 + 2*stride_r1, r02)
        tl.store(R_ptr + rb + 1*stride_r0 + 0*stride_r1, r10)
        tl.store(R_ptr + rb + 1*stride_r0 + 1*stride_r1, r11)
        tl.store(R_ptr + rb + 1*stride_r0 + 2*stride_r1, r12)
        tl.store(R_ptr + rb + 2*stride_r0 + 0*stride_r1, r20)
        tl.store(R_ptr + rb + 2*stride_r0 + 1*stride_r1, r21)
        tl.store(R_ptr + rb + 2*stride_r0 + 2*stride_r1, r22)
        tl.store(T_ptr + b*stride_tb + 0*stride_tn, t0)
        tl.store(T_ptr + b*stride_tb + 1*stride_tn, t1)
        tl.store(T_ptr + b*stride_tb + 2*stride_tn, t2)

    @triton.jit
    def _tm_score_fused_kernel(
        P_ptr, Q_ptr, V_ptr, R_ptr, T_ptr, D0SQ_ptr, LNORM_ptr, SD8SQ_ptr,
        OUT_ptr,
        B, N,
        stride_pb, stride_pn, stride_pc,
        stride_qb, stride_qn, stride_qc,
        stride_vb, stride_vn,
        stride_rb, stride_r0, stride_r1,
        stride_tb, stride_tn,
        BLOCK_N: tl.constexpr,
        USE_SCORE_D8: tl.constexpr,
    ):
        """Fused transform + TM-score in a single kernel launch."""
        b = tl.program_id(0)
        if b >= B:
            return
        r00 = tl.load(R_ptr + b * stride_rb + 0 * stride_r0 + 0 * stride_r1).to(tl.float32)
        r01 = tl.load(R_ptr + b * stride_rb + 0 * stride_r0 + 1 * stride_r1).to(tl.float32)
        r02 = tl.load(R_ptr + b * stride_rb + 0 * stride_r0 + 2 * stride_r1).to(tl.float32)
        r10 = tl.load(R_ptr + b * stride_rb + 1 * stride_r0 + 0 * stride_r1).to(tl.float32)
        r11 = tl.load(R_ptr + b * stride_rb + 1 * stride_r0 + 1 * stride_r1).to(tl.float32)
        r12 = tl.load(R_ptr + b * stride_rb + 1 * stride_r0 + 2 * stride_r1).to(tl.float32)
        r20 = tl.load(R_ptr + b * stride_rb + 2 * stride_r0 + 0 * stride_r1).to(tl.float32)
        r21 = tl.load(R_ptr + b * stride_rb + 2 * stride_r0 + 1 * stride_r1).to(tl.float32)
        r22 = tl.load(R_ptr + b * stride_rb + 2 * stride_r0 + 2 * stride_r1).to(tl.float32)
        t0 = tl.load(T_ptr + b * stride_tb + 0 * stride_tn).to(tl.float32)
        t1 = tl.load(T_ptr + b * stride_tb + 1 * stride_tn).to(tl.float32)
        t2 = tl.load(T_ptr + b * stride_tb + 2 * stride_tn).to(tl.float32)
        d0sq = tl.load(D0SQ_ptr + b).to(tl.float32)
        lnorm = tl.load(LNORM_ptr + b).to(tl.float32)
        if USE_SCORE_D8:
            sd8sq = tl.load(SD8SQ_ptr + b).to(tl.float32)
        acc = tl.zeros((), dtype=tl.float32)
        offs_base = tl.arange(0, BLOCK_N)
        for n0 in tl.range(0, N, BLOCK_N):
            offs = n0 + offs_base
            in_bounds = offs < N
            v = tl.load(V_ptr + b * stride_vb + offs * stride_vn, mask=in_bounds, other=0).to(tl.int32)
            p0 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 0 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32)
            p1 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 1 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32)
            p2 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 2 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32)
            q0 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 0 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32)
            q1 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 1 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32)
            q2 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 2 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32)
            m0 = p0 * r00 + p1 * r10 + p2 * r20 + t0
            m1 = p0 * r01 + p1 * r11 + p2 * r21 + t1
            m2 = p0 * r02 + p1 * r12 + p2 * r22 + t2
            dx = m0 - q0; dy = m1 - q1; dz = m2 - q2
            d2 = dx * dx + dy * dy + dz * dz
            term = 1.0 / (1.0 + d2 / tl.maximum(d0sq, 1e-12))
            valid = v != 0
            if USE_SCORE_D8:
                term = tl.where(valid & (d2 <= sd8sq), term, 0.0)
            else:
                term = tl.where(valid, term, 0.0)
            acc += tl.sum(term, axis=0)
        tl.store(OUT_ptr + b, acc / tl.maximum(lnorm, 1.0))

    @triton.jit
    def _dist2_fused_kernel(
        P_ptr, Q_ptr, V_ptr, R_ptr, T_ptr, OUT_ptr,
        B, N,
        stride_pb, stride_pn, stride_pc,
        stride_qb, stride_qn, stride_qc,
        stride_vb, stride_vn,
        stride_rb, stride_r0, stride_r1,
        stride_tb, stride_tn,
        stride_ob, stride_on,
        BLOCK_N: tl.constexpr,
    ):
        """Fused transform + distance²: (P, Q, V, R, t) → d2 (B, N)."""
        b = tl.program_id(0)
        pid_n = tl.program_id(1)
        if b >= B:
            return
        offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        in_bounds = offs < N
        r00 = tl.load(R_ptr + b * stride_rb + 0 * stride_r0 + 0 * stride_r1).to(tl.float32)
        r01 = tl.load(R_ptr + b * stride_rb + 0 * stride_r0 + 1 * stride_r1).to(tl.float32)
        r02 = tl.load(R_ptr + b * stride_rb + 0 * stride_r0 + 2 * stride_r1).to(tl.float32)
        r10 = tl.load(R_ptr + b * stride_rb + 1 * stride_r0 + 0 * stride_r1).to(tl.float32)
        r11 = tl.load(R_ptr + b * stride_rb + 1 * stride_r0 + 1 * stride_r1).to(tl.float32)
        r12 = tl.load(R_ptr + b * stride_rb + 1 * stride_r0 + 2 * stride_r1).to(tl.float32)
        r20 = tl.load(R_ptr + b * stride_rb + 2 * stride_r0 + 0 * stride_r1).to(tl.float32)
        r21 = tl.load(R_ptr + b * stride_rb + 2 * stride_r0 + 1 * stride_r1).to(tl.float32)
        r22 = tl.load(R_ptr + b * stride_rb + 2 * stride_r0 + 2 * stride_r1).to(tl.float32)
        t0 = tl.load(T_ptr + b * stride_tb + 0 * stride_tn).to(tl.float32)
        t1 = tl.load(T_ptr + b * stride_tb + 1 * stride_tn).to(tl.float32)
        t2 = tl.load(T_ptr + b * stride_tb + 2 * stride_tn).to(tl.float32)
        p0 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 0 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32)
        p1 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 1 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32)
        p2 = tl.load(P_ptr + b * stride_pb + offs * stride_pn + 2 * stride_pc, mask=in_bounds, other=0.0).to(tl.float32)
        q0 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 0 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32)
        q1 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 1 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32)
        q2 = tl.load(Q_ptr + b * stride_qb + offs * stride_qn + 2 * stride_qc, mask=in_bounds, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + b * stride_vb + offs * stride_vn, mask=in_bounds, other=0).to(tl.int32)
        m0 = p0 * r00 + p1 * r10 + p2 * r20 + t0
        m1 = p0 * r01 + p1 * r11 + p2 * r21 + t1
        m2 = p0 * r02 + p1 * r12 + p2 * r22 + t2
        dx = m0 - q0; dy = m1 - q1; dz = m2 - q2
        d2 = dx * dx + dy * dy + dz * dz
        d2 = tl.where(v != 0, d2, 1e12)
        tl.store(OUT_ptr + b * stride_ob + offs * stride_on, d2, mask=in_bounds)

def _kabsch_fused_triton(
    P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-launch fused Kabsch: (P, Q, mask) → (R, t)."""
    B, N, _ = P.shape
    R = torch.empty((B, 3, 3), dtype=torch.float32, device=P.device)
    t = torch.empty((B, 3), dtype=torch.float32, device=P.device)
    m_u8 = mask if mask.dtype == torch.uint8 else mask.to(dtype=torch.uint8)
    block_n, num_warps, num_stages = _triton_kabsch_launch_config(P.device, N)
    _kabsch_fused_kernel[(B,)](
        P, Q, m_u8, R, t,
        B, N,
        *P.stride(),
        *Q.stride(),
        *m_u8.stride(),
        *R.stride(),
        *t.stride(),
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return R, t


def _select_mask_triton(
    pred: torch.Tensor,
    native: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    valid_u8: torch.Tensor,
    d0_search_sq: torch.Tensor,
    out_u8: torch.Tensor,
    near_flags_i32: torch.Tensor,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
) -> torch.Tensor:
    B, N, _ = pred.shape
    block_n, num_warps, num_stages = _triton_select_launch_config(pred.device, N)
    grid = (B, triton.cdiv(N, block_n))
    _select_mask_kernel[grid](
        pred, native, R, t, valid_u8, d0_search_sq, out_u8, near_flags_i32,
        B, N,
        *pred.stride(),
        *native.stride(),
        *R.stride(),
        *t.stride(),
        *valid_u8.stride(),
        *out_u8.stride(),
        near_flags_i32.stride(0),
        abs_tol, rel_tol,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_u8


def _can_use_triton_score(pred: torch.Tensor) -> bool:
    # No min-B gate: one program per batch element with a full N-loop,
    # so even B=1 amortises the single kernel launch vs ~7 torch launches.
    return bool(
        _HAS_TRITON
        and rt._ENABLE_TRITON_SCORE
        and pred.is_cuda
        and pred.dtype == torch.float32
        and pred.is_contiguous()
    )


def _triton_score_launch_config(device: torch.device, N: int) -> tuple[int, int, int]:
    if rt._is_blackwell(device):
        if N >= 384:
            return 512, 4, 3
        if N >= 128:
            return 256, 4, 3
        if N >= 64:
            return 128, 2, 2
        return 64, 2, 2
    block_n = 256 if N >= 256 else (128 if N >= 128 else 64)
    return block_n, 4, 2


def _tm_score_fused_triton(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    d0: torch.Tensor,
    Lnorm: torch.Tensor,
    score_d8: torch.Tensor | None,
) -> torch.Tensor:
    """Fused transform + TM-score in a single Triton kernel launch."""
    B, N, _ = pred.shape
    v_u8 = valid if valid.dtype == torch.uint8 else valid.to(dtype=torch.uint8)
    d0_sq = d0 * d0
    out = torch.empty(B, dtype=torch.float32, device=pred.device)
    block_n, num_warps, num_stages = _triton_score_launch_config(pred.device, N)
    use_sd8 = score_d8 is not None
    sd8_sq = (score_d8 * score_d8) if use_sd8 else d0_sq  # dummy, not read
    _tm_score_fused_kernel[(B,)](
        pred, native, v_u8, R, t, d0_sq, Lnorm, sd8_sq, out,
        B, N,
        *pred.stride(),
        *native.stride(),
        *v_u8.stride(),
        *R.stride(),
        *t.stride(),
        BLOCK_N=block_n,
        USE_SCORE_D8=use_sd8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _dist2_fused_triton(
    pred: torch.Tensor,
    native: torch.Tensor,
    valid: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Fused transform + distance²: returns (B, N) with invalid → 1e12."""
    B, N, _ = pred.shape
    v_u8 = valid if valid.dtype == torch.uint8 else valid.to(dtype=torch.uint8)
    d2 = torch.empty(B, N, dtype=torch.float32, device=pred.device)
    block_n, num_warps, num_stages = _triton_score_launch_config(pred.device, N)
    grid = (B, triton.cdiv(N, block_n))
    _dist2_fused_kernel[grid](
        pred, native, v_u8, R, t, d2,
        B, N,
        *pred.stride(),
        *native.stride(),
        *v_u8.stride(),
        *R.stride(),
        *t.stride(),
        *d2.stride(),
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return d2
