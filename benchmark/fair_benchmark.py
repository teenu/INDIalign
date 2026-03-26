#!/usr/bin/env python3
"""Fair benchmark: common TM-score function applied to both tools' R,t.

Both INDIalign and USalign produce a rotation (R) and translation (t).
This script scores both transforms with one identical TM-score function
(same d0, same formula) to eliminate any self-reporting bias.

Usage:
    python fair_benchmark.py [n_pairs]

Environment variables:
    USALIGN    Path to USalign binary   (default: USalign on PATH)
    INDIALIB   Path to libindialign.so  (default: ../indialign_c/libindialign.so)
"""
import ctypes, math, os, re, subprocess, sys, tempfile, time
import numpy as np
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USALIGN = os.environ.get("USALIGN", "USalign")
INDIALIB = os.environ.get("INDIALIB",
    os.path.join(SCRIPT_DIR, "..", "indialign_c", "libindialign.so"))
lib = ctypes.CDLL(INDIALIB)

# ── ctypes structs ──
class Cfg(ctypes.Structure):
    _fields_ = [("backend_mode",ctypes.c_int),("use_fragment_search",ctypes.c_int),
                ("max_iter",ctypes.c_int),("dp_iter",ctypes.c_int),
                ("max_mem_gb",ctypes.c_double),("request_cuda",ctypes.c_int),
                ("rescue_score_1",ctypes.c_double),("rescue_score_2",ctypes.c_double),
                ("early_term_score",ctypes.c_double)]
class Inp(ctypes.Structure):
    _fields_ = [("length",ctypes.c_int),
                ("pred_xyz",ctypes.POINTER(ctypes.c_double)),
                ("native_xyz",ctypes.POINTER(ctypes.c_double)),
                ("valid_mask",ctypes.POINTER(ctypes.c_uint8)),
                ("pred_valid_mask",ctypes.POINTER(ctypes.c_uint8)),
                ("native_valid_mask",ctypes.POINTER(ctypes.c_uint8)),
                ("lnorm",ctypes.c_double)]
class Res(ctypes.Structure):
    _fields_ = [("score",ctypes.c_double),("R",ctypes.c_double*9),("t",ctypes.c_double*3)]

lib.indialign_default_config.argtypes = [ctypes.POINTER(Cfg)]
lib.indialign_default_config.restype = None
lib.indialign_tmscore_search.argtypes = [ctypes.POINTER(Inp),ctypes.POINTER(Cfg),ctypes.POINTER(Res)]
lib.indialign_tmscore_search.restype = ctypes.c_int

# ── Common d0 (identical in both tools for RNA) ──
def d0_rna(L):
    if   L <= 11: return 0.3
    elif L <= 15: return 0.4
    elif L <= 19: return 0.5
    elif L <= 23: return 0.6
    elif L <  30: return 0.7
    else:         return 0.6 * math.sqrt(L - 0.5) - 2.5

# ── Common scorer: TM-score from R,t ──
def common_tm(pred, native, R, t, d0, Lnorm):
    transformed = pred @ R.T + t
    d2 = np.sum((transformed - native)**2, axis=1)
    return np.sum(1.0 / (1.0 + d2 / (d0 * d0))) / Lnorm

# ── INDIalign: returns (score, R_3x3, t_3, elapsed) ──
def run_indialign(pred, native):
    N = len(native)
    pf = np.ascontiguousarray(pred.flatten(), dtype=np.float64)
    nf = np.ascontiguousarray(native.flatten(), dtype=np.float64)
    v = np.ones(N, dtype=np.uint8)
    inp = Inp()
    inp.length = N
    inp.pred_xyz = pf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    inp.native_xyz = nf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    inp.valid_mask = v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    inp.pred_valid_mask = v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    inp.native_valid_mask = v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    inp.lnorm = float(N)
    cfg = Cfg(); lib.indialign_default_config(ctypes.byref(cfg))
    res = Res()
    t0 = time.perf_counter()
    lib.indialign_tmscore_search(ctypes.byref(inp), ctypes.byref(cfg), ctypes.byref(res))
    dt = time.perf_counter() - t0
    R_flat = np.array(list(res.R))
    R_mat = R_flat.reshape(3, 3).T  # column-major -> row-major
    t_vec = np.array(list(res.t))
    return res.score, R_mat, t_vec, dt

# ── USalign: returns (score, R_3x3, t_3, elapsed) ──
def write_pdb(coords, path):
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(coords):
            f.write(f"ATOM  {i+1:>5d}  C1'   A A{i+1:>4d}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n")

def run_usalign(pred, native):
    with tempfile.TemporaryDirectory() as d:
        pp, np_ = os.path.join(d,"p.pdb"), os.path.join(d,"n.pdb")
        mf = os.path.join(d, "m.txt")
        write_pdb(pred, pp); write_pdb(native, np_)
        t0 = time.perf_counter()
        out = subprocess.run(
            [USALIGN, pp, np_, "-atom", " C1'", "-TMscore", "1", "-m", mf],
            capture_output=True, text=True, timeout=30)
        dt = time.perf_counter() - t0
        matches = re.findall(r"TM-score=\s+([\d.]+)", out.stdout)
        score = float(matches[1]) if len(matches) >= 2 else 0.0
        R = np.zeros((3,3)); t = np.zeros(3)
        lines = open(mf).readlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 5 and parts[0] in ('0','1','2'):
                m = int(parts[0])
                t[m] = float(parts[1])
                R[m] = [float(parts[2]), float(parts[3]), float(parts[4])]
    return score, R, t, dt

# ── Generate RNA-like coords ──
def make_rna(N, rng):
    coords = np.zeros((N, 3))
    d = rng.randn(3); d /= np.linalg.norm(d)
    for i in range(1, N):
        d = d + rng.randn(3) * 0.3; d /= np.linalg.norm(d)
        coords[i] = coords[i-1] + d * (5.9 + rng.randn() * 0.5)
    return coords - coords.mean(axis=0)

# ── Main ──
n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
rng = np.random.RandomState(42)

rows = []
for idx in range(n_pairs):
    N = rng.randint(40, 301)
    sigma = rng.uniform(0.5, 14.0)
    native = make_rna(N, rng)
    pred = Rotation.random(random_state=rng).apply(native) + rng.randn(N,3)*sigma + rng.randn(3)*20

    sc_i, R_i, t_i, dt_i = run_indialign(pred, native)
    sc_u, R_u, t_u, dt_u = run_usalign(pred, native)

    d0 = d0_rna(N)
    common_i = common_tm(pred, native, R_i, t_i, d0, N)
    common_u = common_tm(pred, native, R_u, t_u, d0, N)

    rows.append(dict(idx=idx, N=N, sigma=sigma,
                     self_i=sc_i, self_u=sc_u,
                     common_i=common_i, common_u=common_u,
                     dt_i=dt_i, dt_u=dt_u))
    if (idx+1) % 10 == 0:
        ci = np.mean([r["common_i"] for r in rows])
        cu = np.mean([r["common_u"] for r in rows])
        ti = np.mean([r["dt_i"] for r in rows]) * 1000
        tu = np.mean([r["dt_u"] for r in rows]) * 1000
        wi = sum(1 for r in rows if r["common_i"] - r["common_u"] > 1e-6)
        wu = sum(1 for r in rows if r["common_u"] - r["common_i"] > 1e-6)
        print(f"  [{idx+1:3d}/{n_pairs}]  I wins={wi}  U wins={wu}  "
              f"mean common: I={ci:.4f} U={cu:.4f}  "
              f"speed: I={ti:.0f}ms U={tu:.0f}ms ({ti/tu:.1f}x)")

# ── Analysis ──
n = len(rows)
common_i = np.array([r["common_i"] for r in rows])
common_u = np.array([r["common_u"] for r in rows])
self_i = np.array([r["self_i"] for r in rows])
self_u = np.array([r["self_u"] for r in rows])
dt_i = np.array([r["dt_i"] for r in rows])
dt_u = np.array([r["dt_u"] for r in rows])
sigmas = np.array([r["sigma"] for r in rows])
delta_common = common_i - common_u
delta_self = self_i - self_u

wins_i = int(np.sum(delta_common > 1e-6))
wins_u = int(np.sum(delta_common < -1e-6))
ties = n - wins_i - wins_u

print(f"\n{'='*72}")
print(f"  FAIR COMPARISON: Common TM-score (identical d0) on both R,t")
print(f"{'='*72}")
print(f"  Pairs: {n}")
print(f"  INDIalign wins: {wins_i}  ({100*wins_i/n:.1f}%)")
print(f"  USalign wins:   {wins_u}  ({100*wins_u/n:.1f}%)")
print(f"  Ties:           {ties}  ({100*ties/n:.1f}%)")
print(f"\n  Common scorer (same d0, same formula, both R,t):")
print(f"    Mean INDIalign: {common_i.mean():.6f}")
print(f"    Mean USalign:   {common_u.mean():.6f}")
print(f"    Mean delta:     {delta_common.mean():+.6f}")
print(f"    Median delta:   {np.median(delta_common):+.6f}")
print(f"    Max (I best):   {delta_common.max():+.6f}")
print(f"    Min (U best):   {delta_common.min():+.6f}")
print(f"\n  Self-reported scores (each tool's own scorer):")
print(f"    Mean INDIalign: {self_i.mean():.6f}")
print(f"    Mean USalign:   {self_u.mean():.6f}")
print(f"    Mean delta:     {delta_self.mean():+.6f}")
print(f"\n  Self vs common correlation:")
print(f"    INDIalign self vs common: {np.corrcoef(self_i, common_i)[0,1]:.6f}")
print(f"    USalign self vs common:   {np.corrcoef(self_u, common_u)[0,1]:.6f}")
print(f"    Max |self-common| INDI:   {np.max(np.abs(self_i - common_i)):.6f}")
print(f"    Max |self-common| USal:   {np.max(np.abs(self_u - common_u)):.6f}")

# Speed
print(f"\n  Speed:")
print(f"    INDIalign mean: {dt_i.mean()*1000:.1f}ms/pair  total: {dt_i.sum():.1f}s")
print(f"    USalign mean:   {dt_u.mean()*1000:.1f}ms/pair  total: {dt_u.sum():.1f}s")
print(f"    Ratio (I/U):    {dt_i.mean()/dt_u.mean():.1f}x slower")

# Statistical tests
from scipy.stats import binomtest, wilcoxon
nontie = delta_common[np.abs(delta_common) > 1e-6]
pos = int(np.sum(nontie > 0)); neg = int(np.sum(nontie < 0))
if pos + neg > 0:
    bt = binomtest(pos, pos+neg, 0.5, alternative="greater")
    print(f"\n  Sign test p-value: {bt.pvalue:.4e}  ({pos} vs {neg})")
if len(nontie) >= 10:
    w, wp = wilcoxon(nontie, zero_method="wilcox", alternative="greater")
    print(f"  Wilcoxon p-value:  {wp:.4e}")

# Stratified by difficulty
tm_avg = (common_i + common_u) / 2
bins = [(0,0.17,"very hard (TM<0.17)"), (0.17,0.3,"hard (0.17-0.3)"),
        (0.3,0.5,"medium (0.3-0.5)"), (0.5,0.7,"moderate (0.5-0.7)"),
        (0.7,1.01,"easy (TM>0.7)")]
print(f"\n  {'Stratum':<24s} {'N':>5s} {'I>':>5s} {'U>':>5s} {'Tie':>5s} {'Mean delta':>11s}")
print(f"  {'-'*24} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*11}")
for lo, hi, name in bins:
    mask = (tm_avg >= lo) & (tm_avg < hi)
    if mask.sum() == 0: continue
    d = delta_common[mask]
    iw = int(np.sum(d > 1e-6)); uw = int(np.sum(d < -1e-6))
    ti = int(mask.sum()) - iw - uw
    print(f"  {name:<24s} {int(mask.sum()):5d} {iw:5d} {uw:5d} {ti:5d} {d.mean():+11.6f}")
