#!/usr/bin/env python3
"""Benchmark INDIalign vs USalign on real PDB RNA structures.

Downloads NMR RNA structures from RCSB PDB, extracts C1' coordinates
from multiple models, and benchmarks both tools on model-vs-model pairs.
"""
import ctypes, json, math, os, re, subprocess, sys, tempfile, time, urllib.request
import numpy as np

USALIGN = os.environ.get("USALIGN", "/tmp/USalign/USalign")
INDIALIB = os.environ.get("INDIALIB", "/tmp/INDIalign/indialign_c/libindialign.so")
lib = ctypes.CDLL(INDIALIB)

# ── ctypes structs ──
class Cfg(ctypes.Structure):
    _fields_ = [("backend_mode",ctypes.c_int),("use_fragment_search",ctypes.c_int),
                ("max_iter",ctypes.c_int),("dp_iter",ctypes.c_int),
                ("max_mem_gb",ctypes.c_double),("request_cuda",ctypes.c_int)]
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

def d0_rna(L):
    if   L <= 11: return 0.3
    elif L <= 15: return 0.4
    elif L <= 19: return 0.5
    elif L <= 23: return 0.6
    elif L <  30: return 0.7
    else:         return 0.6 * math.sqrt(L - 0.5) - 2.5

def common_tm(pred, native, R, t, d0, Lnorm):
    transformed = pred @ R.T + t
    d2 = np.sum((transformed - native)**2, axis=1)
    return np.sum(1.0 / (1.0 + d2 / (d0 * d0))) / Lnorm

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
    R_mat = R_flat.reshape(3, 3).T
    t_vec = np.array(list(res.t))
    return res.score, R_mat, t_vec, dt

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

# ── PDB download and C1' extraction ──

def download_pdb(pdb_id):
    """Download PDB file, return text content."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "INDIalign-benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        print(f"  Failed to download {pdb_id}: {e}")
        return None

def extract_c1_models(pdb_text):
    """Extract C1' coordinates per model from PDB text.
    Returns list of numpy arrays, one per model. Each is (N, 3)."""
    models = []
    current = []
    in_model = False
    for line in pdb_text.splitlines():
        rec = line[:6].strip()
        if rec == "MODEL":
            in_model = True
            current = []
        elif rec == "ENDMDL":
            if current:
                models.append(np.array(current))
            current = []
        elif rec == "ATOM":
            name = line[12:16].strip()
            if name == "C1'":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current.append([x, y, z])
    # Single-model structure (no MODEL/ENDMDL)
    if not models and current:
        models.append(np.array(current))
    return models

def search_rna_nmr_pdb_ids():
    """Search RCSB for NMR RNA-only structures, 30-300 nt."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "exptl.method",
                                "operator": "exact_match",
                                "value": "SOLUTION NMR"}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                                "operator": "greater_or_equal", "value": 1}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein",
                                "operator": "equals", "value": 0}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                                "operator": "range",
                                "value": {"from": 30, "to": 300,
                                          "include_lower": True, "include_upper": True}}}
            ]
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 100}}
    }
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    data = json.dumps(query).encode()
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json",
                                         "User-Agent": "INDIalign-benchmark/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        ids = [hit["identifier"] for hit in result.get("result_set", [])]
        print(f"  RCSB search returned {len(ids)} RNA NMR structures")
        return ids
    except Exception as e:
        print(f"  RCSB search failed: {e}")
        return []

# ── Main ──

print("Searching RCSB PDB for RNA NMR structures...")
pdb_ids = search_rna_nmr_pdb_ids()
if not pdb_ids:
    print("No structures found. Exiting.")
    sys.exit(1)

max_structures = int(sys.argv[1]) if len(sys.argv) > 1 else 40
pdb_ids = pdb_ids[:max_structures]

# Download and extract models
structures = {}
for pid in pdb_ids:
    text = download_pdb(pid)
    if text is None:
        continue
    models = extract_c1_models(text)
    if len(models) < 2:
        continue
    # All models must have same length
    lens = set(len(m) for m in models)
    if len(lens) != 1:
        continue
    N = lens.pop()
    if N < 20 or N > 4096:
        continue
    structures[pid] = models
    print(f"  {pid}: {len(models)} models, {N} residues")

print(f"\nUsable structures: {len(structures)}")
if len(structures) < 3:
    print("Too few structures. Exiting.")
    sys.exit(1)

# Build pairs: for each structure, pair model 0 vs models at varying distances
rows = []
pair_idx = 0
for pid, models in structures.items():
    N = len(models[0])
    # Pick model indices spread across the ensemble
    n_models = len(models)
    if n_models <= 5:
        test_models = list(range(1, n_models))
    else:
        test_models = [1, n_models // 4, n_models // 2, 3 * n_models // 4, n_models - 1]
        test_models = sorted(set(test_models))

    native = models[0]
    for mi in test_models:
        pred = models[mi]
        sc_i, R_i, t_i, dt_i = run_indialign(pred, native)
        sc_u, R_u, t_u, dt_u = run_usalign(pred, native)

        d0 = d0_rna(N)
        common_i = common_tm(pred, native, R_i, t_i, d0, N)
        common_u = common_tm(pred, native, R_u, t_u, d0, N)

        rows.append(dict(idx=pair_idx, pdb=pid, model=mi, N=N,
                         self_i=sc_i, self_u=sc_u,
                         common_i=common_i, common_u=common_u,
                         dt_i=dt_i, dt_u=dt_u))
        pair_idx += 1

    if pair_idx % 10 < len(test_models):
        ci = np.mean([r["common_i"] for r in rows])
        cu = np.mean([r["common_u"] for r in rows])
        wi = sum(1 for r in rows if r["common_i"] - r["common_u"] > 1e-6)
        wu = sum(1 for r in rows if r["common_u"] - r["common_i"] > 1e-6)
        print(f"  [{pair_idx:3d} pairs]  I wins={wi}  U wins={wu}  "
              f"mean common: I={ci:.4f} U={cu:.4f}")

# ── Analysis ──
n = len(rows)
if n == 0:
    print("No pairs generated. Exiting.")
    sys.exit(1)

common_i = np.array([r["common_i"] for r in rows])
common_u = np.array([r["common_u"] for r in rows])
self_i = np.array([r["self_i"] for r in rows])
self_u = np.array([r["self_u"] for r in rows])
dt_i = np.array([r["dt_i"] for r in rows])
dt_u = np.array([r["dt_u"] for r in rows])
delta = common_i - common_u

wins_i = int(np.sum(delta > 1e-6))
wins_u = int(np.sum(delta < -1e-6))
ties = n - wins_i - wins_u

print(f"\n{'='*72}")
print(f"  REAL PDB BENCHMARK: NMR RNA model-vs-model pairs")
print(f"{'='*72}")
print(f"  Structures: {len(structures)}")
print(f"  Pairs: {n}")
print(f"  INDIalign wins: {wins_i}  ({100*wins_i/n:.1f}%)")
print(f"  USalign wins:   {wins_u}  ({100*wins_u/n:.1f}%)")
print(f"  Ties:           {ties}  ({100*ties/n:.1f}%)")
print(f"\n  Common scorer:")
print(f"    Mean INDIalign: {common_i.mean():.6f}")
print(f"    Mean USalign:   {common_u.mean():.6f}")
print(f"    Mean delta:     {delta.mean():+.6f}")
print(f"    Median delta:   {np.median(delta):+.6f}")
print(f"    Max (I best):   {delta.max():+.6f}")
print(f"    Min (U best):   {delta.min():+.6f}")
print(f"\n  Self vs common correlation:")
print(f"    INDIalign: {np.corrcoef(self_i, common_i)[0,1]:.6f}")
print(f"    USalign:   {np.corrcoef(self_u, common_u)[0,1]:.6f}")
print(f"\n  Speed:")
print(f"    INDIalign mean: {dt_i.mean()*1000:.1f}ms  total: {dt_i.sum():.1f}s")
print(f"    USalign mean:   {dt_u.mean()*1000:.1f}ms  total: {dt_u.sum():.1f}s")
print(f"    Ratio (I/U):    {dt_i.mean()/dt_u.mean():.1f}x")

from scipy.stats import binomtest, wilcoxon
nontie = delta[np.abs(delta) > 1e-6]
pos = int(np.sum(nontie > 0)); neg = int(np.sum(nontie < 0))
if pos + neg > 0:
    bt = binomtest(pos, pos+neg, 0.5, alternative="greater")
    print(f"\n  Sign test p-value: {bt.pvalue:.4e}  ({pos} vs {neg})")
if len(nontie) >= 10:
    w, wp = wilcoxon(nontie, zero_method="wilcox", alternative="greater")
    print(f"  Wilcoxon p-value:  {wp:.4e}")

# Stratified
tm_avg = (common_i + common_u) / 2
bins = [(0,0.3,"hard (TM<0.3)"), (0.3,0.5,"medium (0.3-0.5)"),
        (0.5,0.7,"moderate (0.5-0.7)"), (0.7,1.01,"easy (TM>0.7)")]
print(f"\n  {'Stratum':<24s} {'N':>5s} {'I>':>5s} {'U>':>5s} {'Tie':>5s} {'Mean delta':>11s}")
print(f"  {'-'*24} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*11}")
for lo, hi, name in bins:
    mask = (tm_avg >= lo) & (tm_avg < hi)
    if mask.sum() == 0: continue
    d = delta[mask]
    iw = int(np.sum(d > 1e-6)); uw = int(np.sum(d < -1e-6))
    ti = int(mask.sum()) - iw - uw
    print(f"  {name:<24s} {int(mask.sum()):5d} {iw:5d} {uw:5d} {ti:5d} {d.mean():+11.6f}")

# Per-structure breakdown
print(f"\n  Per-structure results:")
print(f"  {'PDB':<6s} {'N':>4s} {'Pairs':>5s} {'I>':>4s} {'U>':>4s} {'Mean delta':>11s} {'Avg TM':>7s}")
print(f"  {'-'*6} {'-'*4} {'-'*5} {'-'*4} {'-'*4} {'-'*11} {'-'*7}")
for pid in structures:
    pr = [r for r in rows if r["pdb"] == pid]
    if not pr: continue
    di = np.array([r["common_i"] - r["common_u"] for r in pr])
    avg_tm = np.mean([(r["common_i"] + r["common_u"])/2 for r in pr])
    iw = int(np.sum(di > 1e-6)); uw = int(np.sum(di < -1e-6))
    print(f"  {pid:<6s} {pr[0]['N']:4d} {len(pr):5d} {iw:4d} {uw:4d} {di.mean():+11.6f} {avg_tm:7.4f}")
