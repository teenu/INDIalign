# INDIalign

RNA structural alignment via multi-strategy TM-score optimization.

INDIalign finds the rigid-body superposition (rotation **R**, translation **t**) that maximizes the RNA TM-score between two sets of C1' coordinates. It uses a multi-strategy seed-and-refine pipeline that searches more broadly than a single alignment heuristic. On a 500-pair synthetic benchmark scored by an identical common TM-score function, INDIalign finds a higher-scoring superposition on **66% of pairs** (vs 27% for USalign), with the advantage concentrated on hard targets. It is **3.8x slower** than USalign.

## Quick Start

```bash
cd indialign_c && make    # auto-detects CUDA; use `make cpu` for CPU-only
```

### Python (ctypes)

```python
import ctypes, numpy as np

lib = ctypes.CDLL("indialign_c/libindialign.so")

class NativeConfig(ctypes.Structure):
    _fields_ = [("backend_mode", ctypes.c_int),
                ("use_fragment_search", ctypes.c_int),
                ("max_iter", ctypes.c_int), ("dp_iter", ctypes.c_int),
                ("max_mem_gb", ctypes.c_double),
                ("request_cuda", ctypes.c_int)]

class NativePairInput(ctypes.Structure):
    _fields_ = [("length", ctypes.c_int),
                ("pred_xyz", ctypes.POINTER(ctypes.c_double)),
                ("native_xyz", ctypes.POINTER(ctypes.c_double)),
                ("valid_mask", ctypes.POINTER(ctypes.c_uint8)),
                ("pred_valid_mask", ctypes.POINTER(ctypes.c_uint8)),
                ("native_valid_mask", ctypes.POINTER(ctypes.c_uint8)),
                ("lnorm", ctypes.c_double)]

class NativePairResult(ctypes.Structure):
    _fields_ = [("score", ctypes.c_double),
                ("rotation", ctypes.c_double * 9),
                ("translation", ctypes.c_double * 3)]

# Set up input
N = len(pred_coords)  # number of residues
pred = np.ascontiguousarray(pred_coords.flatten(), dtype=np.float64)
native = np.ascontiguousarray(native_coords.flatten(), dtype=np.float64)
valid = np.ones(N, dtype=np.uint8)

inp = NativePairInput()
inp.length = N
inp.pred_xyz = pred.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
inp.native_xyz = native.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
inp.valid_mask = valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
inp.pred_valid_mask = valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
inp.native_valid_mask = valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
inp.lnorm = float(N)

cfg = NativeConfig()
lib.indialign_default_config(ctypes.byref(cfg))

res = NativePairResult()
lib.indialign_tmscore_search(ctypes.byref(inp), ctypes.byref(cfg), ctypes.byref(res))

print(f"TM-score: {res.score:.4f}")
```

## Benchmark: INDIalign vs USalign

### Methodology

Both tools receive the same input pair and each produces its own superposition (R, t). Both transforms are then scored by an **identical common scorer** -- same d0 formula, same TM-score formula `sum(1 / (1 + d^2/d0^2)) / Lnorm`, no d8 filter. This eliminates self-reporting bias.

- 500 synthetic RNA-like coordinate pairs (C1' atoms)
- Lengths: 40--300 residues (uniform random)
- Noise: random rotation + Gaussian perturbation, sigma ~ U(0.5, 14.0) angstroms
- TM-scores span 0.05 to 0.95 across the dataset
- Fixed random seed (42) for reproducibility
- Integrity check: self-reported vs common score correlation = 1.000 for both tools

### Results

| Metric | Value |
|---|---|
| INDIalign wins | 330 / 500 (66.0%) |
| USalign wins | 137 / 500 (27.4%) |
| Ties | 33 / 500 (6.6%) |
| Mean TM-score delta | +0.000477 |
| Median TM-score delta | +0.000103 |
| Sign test p-value | 9.7 x 10^-20 |
| Wilcoxon p-value | 2.1 x 10^-22 |

### Stratified by Difficulty

| Difficulty | N | INDIalign wins | USalign wins | Ties | Mean delta |
|---|---|---|---|---|---|
| Very hard (TM < 0.17) | 168 | 106 | 41 | 21 | +0.000725 |
| Hard (0.17--0.3) | 132 | 101 | 28 | 3 | +0.000657 |
| Medium (0.3--0.5) | 98 | 72 | 26 | 0 | +0.000306 |
| Moderate (0.5--0.7) | 50 | 18 | 31 | 1 | -0.000013 |
| Easy (TM > 0.7) | 52 | 33 | 11 | 8 | +0.000012 |

INDIalign's advantage concentrates on hard and very hard targets. On moderate targets (TM 0.5--0.7), USalign finds slightly better superpositions on average.

### Speed

| Tool | Mean time / pair | Ratio |
|---|---|---|
| USalign | 22 ms | 1.0x |
| INDIalign | 83 ms | 3.8x |

INDIalign trades speed for search breadth. If throughput is the primary constraint, USalign is the better choice.

### Reproducing

```bash
# Requires: USalign binary on PATH (or set USALIGN=/path/to/USalign),
#           numpy, scipy
cd benchmark
python fair_benchmark.py 500
```

## Algorithm

INDIalign runs a multi-stage pipeline. Each stage produces candidate (R, t) superpositions; the best is kept.

| Stage | Source | Description |
|---|---|---|
| 1. Seed generation | `seeds.h` | Fragment seeds at multiple lengths + transform-independent anchor contact seeds |
| 2. Seed bank evaluation | `search.h` | K seeds x C d0-candidates, each refined by iterative Kabsch with convergence detection |
| 3. SS-based alignment | `ss.h` | Secondary structure from C1' geometry, NW alignment (pure SS + combined SS/structure) |
| 4. Gapless threading | `threading.h` | Offset alignments pred[i] -> native[i-k], Kabsch + iterative refinement |
| 5. Rescue strategies | `rescue.h` | Local fragment DP + threading DP, activated at score < 0.5 and < 0.3 |
| 6. Distance seed | `seeds.h` | Top-50% closest residues under current best (R,t), re-evaluated as seed |
| 7. DP refinement | `align.h` | Needleman-Wunsch cross-index DP, Kabsch on aligned pairs, rescored same-index |
| 8. Weighted refinement | `core.h` | 4 iterations: weight residues by TM contribution, weighted Kabsch |

**Core math:** Horn quaternion method via Jacobi 4x4 eigensolve. RNA d0 formula: piecewise for L < 30, `0.6 * sqrt(L - 0.5) - 2.5` for L >= 30 (identical to USalign).

### Key Design Decisions

- **Anchor contact seeds** (`seeds.h`): Identify residue pairs with consistent inter-residue distances in both structures, without needing an initial (R, t). Effective starting points for hard cases.
- **Cross-index DP with same-index rescoring**: DP alignment finds pred[i] -> native[j] mappings. The resulting (R, t) is always rescored with same-index `tm_score_no_d8` on the full coordinate arrays to maintain scoring integrity.
- **Progressive rescue**: Rescue strategies activate only when needed (score < 0.5, then < 0.3), avoiding wasted compute on easy targets. This is the primary source of both the accuracy advantage and the speed cost.
- **Fused GPU kernel** (`gpu_kernels.cu`): A single CUDA kernel performs iterative Kabsch refinement + TM-score per seed, using warp-shuffle reductions and shared-memory eigensolve.

## Code Structure

```
indialign_c/             C++20 core (~1,500 lines C++, ~330 lines CUDA)
  indialign.cpp          Main search pipeline + C API
  api.h                  Public structs: NativePairInput, NativePairResult, NativeConfig
  core.h                 Kabsch (Horn quaternion), TM-score, weighted refinement, d0
  seeds.h                Fragment seeds, anchor contact seeds, top-k distance seeds
  search.h               Seed bank evaluation (OpenMP parallel)
  align.h                Cross-index DP refinement, detailed search on aligned pairs
  nw.h                   Needleman-Wunsch DP with affine gap penalty
  ss.h                   RNA secondary structure from C1' geometry
  threading.h            Gapless threading search
  rescue.h               Local fragment DP, threading DP rescue strategies
  gpu_kernels.cu         Fused CUDA kernel: seed refinement + TM-score
  gpu_device.cuh         Device-side Jacobi eigensolve, warp reductions
  Makefile               Builds libindialign.so (CPU or CPU+CUDA)

scoring/                 Python GPU layer (PyTorch + Triton, for batch evaluation)
benchmark/               Reproducible benchmark vs USalign
```

## Building

```bash
# CPU only (requires g++ with C++20 and OpenMP)
cd indialign_c && make cpu

# With CUDA (auto-detects GPU architecture)
cd indialign_c && make

# Specific CUDA architecture
cd indialign_c && make gpu CUDA_ARCH=sm_90
```

**Requirements:** C++20 compiler (g++ 10+ or clang 13+), OpenMP. Optional: CUDA toolkit 12+ for GPU acceleration. The GPU path uses float32 internally and falls back to CPU on any CUDA error.

**macOS:** Apple Clang does not ship with OpenMP. Install it via `brew install libomp` and add `-I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib` to CXXFLAGS/LDFLAGS, or use GCC (`brew install gcc`).

## C API

```c
#include "api.h"

// Initialize default configuration
void indialign_default_config(NativeConfig *cfg);

// Align a single pair, returns 0 on success
int indialign_tmscore_search(const NativePairInput *inp,
                             const NativeConfig *cfg,
                             NativePairResult *out);

// Align multiple pairs in parallel (OpenMP)
int indialign_tmscore_search_batch(int n_pairs,
                                    const NativePairInput *inputs,
                                    const NativeConfig *cfg,
                                    NativePairResult *outputs);

// Check CUDA availability (1 = available, 0 = not)
int indialign_cuda_available(void);
```

**Input:** `pred_xyz` and `native_xyz` are flat arrays of length `3*N` (x,y,z interleaved). `valid_mask` marks residues present in both structures. `lnorm` is the normalization length (typically N).

**Output:** `rotation` is a 3x3 matrix stored column-major (9 doubles). `translation` is a 3-vector. The transform applies as: `aligned = pred @ R^T + t`.

## Limitations

1. **Speed.** 3.8x slower than USalign on average. The multi-strategy pipeline is inherently more expensive.
2. **Synthetic benchmark.** Current benchmark uses synthetic coordinate pairs with Gaussian noise. Real RNA structures from PDB or prediction models may have different error distributions.
3. **Marginal mean improvement.** The mean TM-score delta is +0.0005. The advantage is in win rate on hard targets, not in average score magnitude.
4. **Max length.** Fixed 4096-element arrays in the C core. Sequences longer than 4096 residues are rejected.
5. **Same-length assumption.** The C API requires pred and native to have the same length N. Structures with different lengths require pre-alignment or padding with invalid masks.

## Citation

If you use INDIalign in your research, please cite:

```bibtex
@software{chalapati2026indialign,
  author  = {Chalapati, Sachin},
  title   = {{INDIalign}: {RNA} structural alignment via multi-strategy
             {TM}-score optimization},
  year    = {2026},
  url     = {https://github.com/teenu/INDIalign},
  version = {1.0.0}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

[USalign](https://github.com/pylelab/USalign) (Zhang & Skolnick lab) is the baseline for comparison. INDIalign uses the same RNA d0 normalization formula as USalign.
