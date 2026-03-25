#!/usr/bin/env python3
"""Benchmark on longer and harder PDB RNA structures.

1. NMR ensembles 50-500 nt: pairs most divergent models (max RMSD)
2. X-ray RNA with multiple identical chains: chain A vs chain B
3. Cross-structure pairs: same Rfam family, different PDB entries
"""
import ctypes, json, math, os, re, subprocess, sys, tempfile, time, urllib.request
import numpy as np

USALIGN = os.environ.get("USALIGN", "/tmp/USalign/USalign")
INDIALIB = os.environ.get("INDIALIB", "/tmp/INDIalign/indialign_c/libindialign.so")
lib = ctypes.CDLL(INDIALIB)

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

def run_indialign(pred, native, valid_p=None, valid_n=None):
    N = len(native)
    pf = np.ascontiguousarray(pred.flatten(), dtype=np.float64)
    nf = np.ascontiguousarray(native.flatten(), dtype=np.float64)
    if valid_p is None: valid_p = np.ones(N, dtype=np.uint8)
    if valid_n is None: valid_n = np.ones(N, dtype=np.uint8)
    valid = (valid_p & valid_n).astype(np.uint8)
    inp = Inp()
    inp.length = N
    inp.pred_xyz = pf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    inp.native_xyz = nf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    inp.valid_mask = valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    inp.pred_valid_mask = valid_p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    inp.native_valid_mask = valid_n.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    Lnorm = float(int(valid.sum()))
    inp.lnorm = max(Lnorm, 1.0)
    cfg = Cfg(); lib.indialign_default_config(ctypes.byref(cfg))
    res = Res()
    t0 = time.perf_counter()
    lib.indialign_tmscore_search(ctypes.byref(inp), ctypes.byref(cfg), ctypes.byref(res))
    dt = time.perf_counter() - t0
    R_flat = np.array(list(res.R))
    R_mat = R_flat.reshape(3, 3).T
    t_vec = np.array(list(res.t))
    return res.score, R_mat, t_vec, dt

def write_pdb(coords, path, valid=None):
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(coords):
            if valid is not None and not valid[i]:
                continue
            f.write(f"ATOM  {i+1:>5d}  C1'   A A{i+1:>4d}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n")

def run_usalign(pred, native, valid_p=None, valid_n=None):
    N = len(native)
    if valid_p is None: valid_p = np.ones(N, dtype=np.uint8)
    if valid_n is None: valid_n = np.ones(N, dtype=np.uint8)
    Lnorm = int((valid_p & valid_n).sum())
    with tempfile.TemporaryDirectory() as d:
        pp, np_ = os.path.join(d,"p.pdb"), os.path.join(d,"n.pdb")
        mf = os.path.join(d, "m.txt")
        write_pdb(pred, pp, valid_p); write_pdb(native, np_, valid_n)
        t0 = time.perf_counter()
        out = subprocess.run(
            [USALIGN, pp, np_, "-atom", " C1'", "-TMscore", "1", "-m", mf],
            capture_output=True, text=True, timeout=60)
        dt = time.perf_counter() - t0
        matches = re.findall(r"TM-score=\s+([\d.]+)", out.stdout)
        score = float(matches[1]) if len(matches) >= 2 else 0.0
        R = np.zeros((3,3)); t = np.zeros(3)
        try:
            lines = open(mf).readlines()
            for line in lines:
                parts = line.split()
                if len(parts) == 5 and parts[0] in ('0','1','2'):
                    m = int(parts[0])
                    t[m] = float(parts[1])
                    R[m] = [float(parts[2]), float(parts[3]), float(parts[4])]
        except: pass
    return score, R, t, dt

def common_tm_masked(pred, native, valid, R, t, d0):
    Lnorm = int(valid.sum())
    if Lnorm == 0: return 0.0
    transformed = pred @ R.T + t
    d2 = np.sum((transformed - native)**2, axis=1)
    scores = valid / (1.0 + d2 / (d0 * d0))
    return np.sum(scores) / Lnorm

# ── PDB utilities ──

def download_pdb(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "INDIalign-benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        print(f"  Failed to download {pdb_id}: {e}")
        return None

def extract_c1_models(pdb_text):
    """Extract C1' coords per model. Returns list of (N,3) arrays."""
    models = []
    current = []
    for line in pdb_text.splitlines():
        rec = line[:6].strip()
        if rec == "MODEL":
            current = []
        elif rec == "ENDMDL":
            if current: models.append(np.array(current))
            current = []
        elif rec == "ATOM":
            if line[12:16].strip() == "C1'":
                current.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not models and current:
        models.append(np.array(current))
    return models

def extract_c1_by_chain(pdb_text):
    """Extract C1' coords per chain from first model. Returns {chain: (N,3)}."""
    chains = {}
    past_first_model = False
    for line in pdb_text.splitlines():
        rec = line[:6].strip()
        if rec == "ENDMDL":
            past_first_model = True
        if past_first_model:
            continue
        if rec == "ATOM" and line[12:16].strip() == "C1'":
            ch = line[21]
            chains.setdefault(ch, []).append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return {ch: np.array(coords) for ch, coords in chains.items()}

def extract_c1_by_chain_resnum(pdb_text):
    """Extract C1' coords with residue numbers per chain. Returns {chain: {resnum: xyz}}."""
    chains = {}
    past_first_model = False
    for line in pdb_text.splitlines():
        rec = line[:6].strip()
        if rec == "ENDMDL":
            past_first_model = True
        if past_first_model:
            continue
        if rec == "ATOM" and line[12:16].strip() == "C1'":
            ch = line[21]
            resnum = int(line[22:26].strip())
            chains.setdefault(ch, {})[resnum] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return chains

def rcsb_search(query):
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    data = json.dumps(query).encode()
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json",
                                         "User-Agent": "INDIalign-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
    return [hit["identifier"] for hit in result.get("result_set", [])]

def most_divergent_pair(models):
    """Find the pair of models with highest C1' RMSD."""
    n = len(models)
    best_rmsd, best_i, best_j = -1, 0, 1
    # Sample if too many models
    indices = list(range(n))
    if n > 20:
        indices = [0] + list(np.linspace(1, n-1, 19, dtype=int))
    for ii, i in enumerate(indices):
        for j in indices[ii+1:]:
            rmsd = np.sqrt(np.mean(np.sum((models[i] - models[j])**2, axis=1)))
            if rmsd > best_rmsd:
                best_rmsd, best_i, best_j = rmsd, i, j
    return best_i, best_j, best_rmsd

def align_by_resnum(chain_a, chain_b):
    """Align two {resnum: xyz} dicts. Returns pred, native, valid arrays."""
    all_res = sorted(set(chain_a.keys()) | set(chain_b.keys()))
    N = len(all_res)
    pred = np.zeros((N, 3))
    native = np.zeros((N, 3))
    vp = np.zeros(N, dtype=np.uint8)
    vn = np.zeros(N, dtype=np.uint8)
    for idx, r in enumerate(all_res):
        if r in chain_a:
            pred[idx] = chain_a[r]; vp[idx] = 1
        if r in chain_b:
            native[idx] = chain_b[r]; vn[idx] = 1
    return pred, native, vp, vn

# ── Pair collection ──

pairs = []  # list of dicts: {label, pred, native, vp, vn, N, source}

# === 1. Large NMR ensembles: most divergent model pair ===
print("=== Phase 1: NMR ensembles (50-500 nt), most divergent pair ===")
try:
    nmr_ids = rcsb_search({
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "exptl.method", "operator": "exact_match",
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
                            "value": {"from": 50, "to": 500}}}
        ]},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 80}}
    })
    print(f"  Found {len(nmr_ids)} NMR structures")
except Exception as e:
    print(f"  NMR search failed: {e}")
    nmr_ids = []

for pid in nmr_ids:
    text = download_pdb(pid)
    if not text: continue
    models = extract_c1_models(text)
    if len(models) < 2: continue
    lens = set(len(m) for m in models)
    if len(lens) != 1: continue
    N = lens.pop()
    if N < 30 or N > 4096: continue
    mi, mj, rmsd = most_divergent_pair(models)
    pred, native = models[mi], models[mj]
    pairs.append(dict(label=f"{pid}_m{mi}v{mj}", pred=pred, native=native,
                      vp=None, vn=None, N=N, source="NMR-divergent",
                      rmsd=rmsd))
    print(f"  {pid}: {N} nt, {len(models)} models, "
          f"most divergent m{mi} vs m{mj} RMSD={rmsd:.1f}A")

# === 2. X-ray RNA with multiple identical chains ===
print("\n=== Phase 2: X-ray RNA, chain-vs-chain ===")
try:
    xray_ids = rcsb_search({
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "exptl.method", "operator": "exact_match",
                            "value": "X-RAY DIFFRACTION"}},
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                            "operator": "greater_or_equal", "value": 2}},
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein",
                            "operator": "equals", "value": 0}},
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                            "operator": "range",
                            "value": {"from": 80, "to": 2000}}}
        ]},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 80}}
    })
    print(f"  Found {len(xray_ids)} X-ray structures")
except Exception as e:
    print(f"  X-ray search failed: {e}")
    xray_ids = []

for pid in xray_ids:
    text = download_pdb(pid)
    if not text: continue
    chains = extract_c1_by_chain(text)
    if len(chains) < 2: continue
    # Find pairs of chains with same length (likely same entity)
    chain_ids = sorted(chains.keys())
    added = False
    for i in range(len(chain_ids)):
        for j in range(i+1, len(chain_ids)):
            ci, cj = chain_ids[i], chain_ids[j]
            if len(chains[ci]) != len(chains[cj]): continue
            N = len(chains[ci])
            if N < 40 or N > 4096: continue
            rmsd = np.sqrt(np.mean(np.sum((chains[ci] - chains[cj])**2, axis=1)))
            pairs.append(dict(label=f"{pid}_{ci}v{cj}", pred=chains[ci],
                              native=chains[cj], vp=None, vn=None, N=N,
                              source="Xray-chain", rmsd=rmsd))
            if not added:
                print(f"  {pid}: chain {ci} vs {cj}, {N} nt, RMSD={rmsd:.1f}A")
                added = True
            break  # one pair per structure
        if added: break

# === 3. Cross-structure pairs: riboswitch families ===
print("\n=== Phase 3: Cross-structure pairs (same sequence family) ===")
# Search for well-studied riboswitch/ribozyme families
family_queries = [
    ("TPP riboswitch", "TPP riboswitch"),
    ("hammerhead ribozyme", "hammerhead ribozyme"),
    ("SAM riboswitch", "SAM riboswitch"),
    ("purine riboswitch", "purine riboswitch"),
    ("HDV ribozyme", "hepatitis delta virus ribozyme"),
]
for family_name, search_term in family_queries:
    try:
        fam_ids = rcsb_search({
            "query": {"type": "group", "logical_operator": "and", "nodes": [
                {"type": "terminal", "service": "full_text",
                 "parameters": {"value": search_term}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                                "operator": "greater_or_equal", "value": 1}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                                "operator": "range",
                                "value": {"from": 40, "to": 1000}}}
            ]},
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 20}}
        })
    except:
        fam_ids = []
    if len(fam_ids) < 2:
        continue
    # Download first few, find longest RNA chain in each, pair by resnum
    fam_chains = []
    for pid in fam_ids[:6]:
        text = download_pdb(pid)
        if not text: continue
        chains = extract_c1_by_chain_resnum(text)
        if not chains: continue
        # Pick longest RNA chain
        best_ch = max(chains, key=lambda c: len(chains[c]))
        if len(chains[best_ch]) < 30: continue
        fam_chains.append((pid, best_ch, chains[best_ch]))
    # Create pairs from first structure vs others
    if len(fam_chains) < 2: continue
    ref_pid, ref_ch, ref_data = fam_chains[0]
    for pid2, ch2, data2 in fam_chains[1:]:
        pred, native, vp, vn = align_by_resnum(data2, ref_data)
        n_valid = int((vp & vn).sum())
        if n_valid < 20: continue
        N = len(pred)
        if N > 4096: continue
        rmsd_valid = []
        for k in range(N):
            if vp[k] and vn[k]:
                rmsd_valid.append(np.sum((pred[k] - native[k])**2))
        rmsd = np.sqrt(np.mean(rmsd_valid)) if rmsd_valid else 0
        pairs.append(dict(label=f"{family_name}:{pid2}v{ref_pid}",
                          pred=pred, native=native, vp=vp, vn=vn,
                          N=N, source=f"cross-{family_name}", rmsd=rmsd))
        print(f"  {family_name}: {pid2} vs {ref_pid}, "
              f"N={N} ({n_valid} shared), RMSD={rmsd:.1f}A")

# ── Run benchmark ──
print(f"\n{'='*72}")
print(f"  Total pairs collected: {len(pairs)}")
print(f"{'='*72}\n")

if not pairs:
    print("No pairs. Exiting.")
    sys.exit(1)

rows = []
for idx, p in enumerate(pairs):
    pred, native = p["pred"], p["native"]
    vp = p["vp"] if p["vp"] is not None else np.ones(p["N"], dtype=np.uint8)
    vn = p["vn"] if p["vn"] is not None else np.ones(p["N"], dtype=np.uint8)
    valid = (vp & vn).astype(np.uint8)
    Lnorm = int(valid.sum())
    if Lnorm < 3: continue

    sc_i, R_i, t_i, dt_i = run_indialign(pred, native, vp, vn)
    sc_u, R_u, t_u, dt_u = run_usalign(pred, native, vp, vn)

    d0 = d0_rna(Lnorm)
    ci = common_tm_masked(pred, native, valid, R_i, t_i, d0)
    cu = common_tm_masked(pred, native, valid, R_u, t_u, d0)

    rows.append(dict(idx=idx, label=p["label"], N=p["N"], Lnorm=Lnorm,
                     source=p["source"], rmsd=p.get("rmsd", 0),
                     common_i=ci, common_u=cu, dt_i=dt_i, dt_u=dt_u))
    delta = ci - cu
    tag = "I>" if delta > 1e-6 else ("U>" if delta < -1e-6 else "==")
    print(f"  [{idx+1:3d}] {tag} {p['label']:<35s} N={p['N']:4d} "
          f"I={ci:.4f} U={cu:.4f} d={delta:+.4f} "
          f"I:{dt_i*1000:.0f}ms U:{dt_u*1000:.0f}ms")

# ── Analysis ──
n = len(rows)
common_i = np.array([r["common_i"] for r in rows])
common_u = np.array([r["common_u"] for r in rows])
dt_i = np.array([r["dt_i"] for r in rows])
dt_u = np.array([r["dt_u"] for r in rows])
Ns = np.array([r["N"] for r in rows])
delta = common_i - common_u

wins_i = int(np.sum(delta > 1e-6))
wins_u = int(np.sum(delta < -1e-6))
ties = n - wins_i - wins_u

print(f"\n{'='*72}")
print(f"  HARD PDB BENCHMARK: longer and challenging RNA pairs")
print(f"{'='*72}")
print(f"  Pairs: {n}  (length range: {int(Ns.min())}-{int(Ns.max())} nt)")
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

# By source
print(f"\n  By source:")
print(f"  {'Source':<25s} {'N':>4s} {'I>':>4s} {'U>':>4s} {'Tie':>4s} {'Mean delta':>11s} {'Avg len':>7s}")
print(f"  {'-'*25} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*11} {'-'*7}")
for src in sorted(set(r["source"] for r in rows)):
    sr = [r for r in rows if r["source"] == src]
    d = np.array([r["common_i"] - r["common_u"] for r in sr])
    iw = int(np.sum(d > 1e-6)); uw = int(np.sum(d < -1e-6))
    avg_n = np.mean([r["N"] for r in sr])
    print(f"  {src:<25s} {len(sr):4d} {iw:4d} {uw:4d} {len(sr)-iw-uw:4d} "
          f"{d.mean():+11.6f} {avg_n:7.0f}")

# By length bin
print(f"\n  By length:")
lbins = [(30,60,"30-60 nt"), (60,100,"60-100 nt"), (100,200,"100-200 nt"),
         (200,500,"200-500 nt"), (500,5000,"500+ nt")]
print(f"  {'Length':<15s} {'N':>4s} {'I>':>4s} {'U>':>4s} {'Mean delta':>11s} {'Speed ratio':>11s}")
print(f"  {'-'*15} {'-'*4} {'-'*4} {'-'*4} {'-'*11} {'-'*11}")
for lo, hi, name in lbins:
    mask = (Ns >= lo) & (Ns < hi)
    if mask.sum() == 0: continue
    d = delta[mask]
    iw = int(np.sum(d > 1e-6)); uw = int(np.sum(d < -1e-6))
    sr = dt_i[mask].mean() / dt_u[mask].mean()
    print(f"  {name:<15s} {int(mask.sum()):4d} {iw:4d} {uw:4d} "
          f"{d.mean():+11.6f} {sr:10.1f}x")

# Stratified by difficulty
tm_avg = (common_i + common_u) / 2
bins = [(0,0.3,"hard (TM<0.3)"), (0.3,0.5,"medium (0.3-0.5)"),
        (0.5,0.7,"moderate (0.5-0.7)"), (0.7,1.01,"easy (TM>0.7)")]
print(f"\n  By difficulty:")
print(f"  {'Stratum':<24s} {'N':>4s} {'I>':>4s} {'U>':>4s} {'Mean delta':>11s}")
print(f"  {'-'*24} {'-'*4} {'-'*4} {'-'*4} {'-'*11}")
for lo, hi, name in bins:
    mask = (tm_avg >= lo) & (tm_avg < hi)
    if mask.sum() == 0: continue
    d = delta[mask]
    iw = int(np.sum(d > 1e-6)); uw = int(np.sum(d < -1e-6))
    print(f"  {name:<24s} {int(mask.sum()):4d} {iw:4d} {uw:4d} {d.mean():+11.6f}")
