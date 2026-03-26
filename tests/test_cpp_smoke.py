#!/usr/bin/env python3
"""Smoke test: verify C++ library works and N>4096 is supported."""
import ctypes, os, sys, math
import numpy as np
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDIALIB = os.environ.get("INDIALIB",
    os.path.join(SCRIPT_DIR, "..", "indialign_c", "libindialign.so"))
lib = ctypes.CDLL(INDIALIB)

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

def make_chain(N, rng):
    coords = np.zeros((N, 3))
    d = rng.randn(3); d /= np.linalg.norm(d)
    for i in range(1, N):
        d = d + rng.randn(3) * 0.3; d /= np.linalg.norm(d)
        coords[i] = coords[i-1] + d * (5.9 + rng.randn() * 0.5)
    return coords - coords.mean(axis=0)

def run(pred, native):
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
    rc = lib.indialign_tmscore_search(ctypes.byref(inp), ctypes.byref(cfg), ctypes.byref(res))
    return rc, res.score

rng = np.random.RandomState(42)
failures = 0

# Test 1: Small easy pair (N=50, low noise)
print("Test 1: N=50, low noise...", end=" ")
native = make_chain(50, rng)
pred = Rotation.random(random_state=rng).apply(native) + rng.randn(50,3)*1.0
rc, score = run(pred, native)
ok = rc == 0 and score > 0.5
print(f"rc={rc} score={score:.4f} {'PASS' if ok else 'FAIL'}")
if not ok: failures += 1

# Test 2: Medium pair (N=200, moderate noise)
print("Test 2: N=200, moderate noise...", end=" ")
native = make_chain(200, rng)
pred = Rotation.random(random_state=rng).apply(native) + rng.randn(200,3)*3.0
rc, score = run(pred, native)
ok = rc == 0 and score > 0.2
print(f"rc={rc} score={score:.4f} {'PASS' if ok else 'FAIL'}")
if not ok: failures += 1

# Test 3: Identity alignment (should give score ~1.0)
print("Test 3: N=100, identity...", end=" ")
native = make_chain(100, rng)
rc, score = run(native.copy(), native)
ok = rc == 0 and score > 0.95
print(f"rc={rc} score={score:.4f} {'PASS' if ok else 'FAIL'}")
if not ok: failures += 1

# Test 4: N > 4096 (the lifted limit)
print("Test 4: N=6000, lifted limit...", end=" ")
native = make_chain(6000, rng)
pred = Rotation.random(random_state=rng).apply(native) + rng.randn(6000,3)*2.0
rc, score = run(pred, native)
ok = rc == 0 and score > 0.3
print(f"rc={rc} score={score:.4f} {'PASS' if ok else 'FAIL'}")
if not ok: failures += 1

# Test 5: N=8192 (well above old limit)
print("Test 5: N=8192, large structure...", end=" ")
native = make_chain(8192, rng)
pred = Rotation.random(random_state=rng).apply(native) + rng.randn(8192,3)*2.0
rc, score = run(pred, native)
ok = rc == 0 and score > 0.3
print(f"rc={rc} score={score:.4f} {'PASS' if ok else 'FAIL'}")
if not ok: failures += 1

print(f"\n{'All tests passed!' if failures == 0 else f'{failures} test(s) FAILED'}")
sys.exit(failures)
