#ifndef INDIALIGN_NW_H
#define INDIALIGN_NW_H

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

/* ── Pairwise TM-score similarity matrix (Np × Nn) ───────────── */

inline void score_matrix(const float *moved, const float *native,
                         const uint8_t *pv, const uint8_t *nv,
                         float d0, float score_d8, int N, float *smat) {
    float d0sq = std::max(d0*d0, 1e-12f), sd8sq = score_d8*score_d8;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float val = 0;
            if (pv[i] && nv[j]) {
                float dx = moved[i*3]  -native[j*3];
                float dy = moved[i*3+1]-native[j*3+1];
                float dz = moved[i*3+2]-native[j*3+2];
                float d2 = dx*dx+dy*dy+dz*dz;
                if (d2 <= sd8sq)
                    val = 1.0f / (1.0f + d2/d0sq);
            }
            smat[i*N+j] = val;
        }
    }
}

/* ── Needleman-Wunsch DP (zero gap penalty) ───────────────────── */

inline void nw_dp(const float *smat, int Np, int Nn,
                  std::vector<float> &H, std::vector<int8_t> &trace) {
    int Hp = Np+1, Hn = Nn+1;
    H.assign(Hp * Hn, 0.0f);
    trace.assign(Hp * Hn, 0);
    for (int d = 2; d < Np + Nn + 2; d++) {
        int ilo = std::max(1, d - Nn);
        int ihi = std::min(Np, d - 1);
        for (int ii = ilo; ii <= ihi; ii++) {
            int jj = d - ii;
            float diag = H[(ii-1)*Hn+(jj-1)] + smat[(ii-1)*Nn+(jj-1)];
            float up   = H[(ii-1)*Hn+jj];
            float left = H[ii*Hn+(jj-1)];
            float best = diag; int8_t dir = 0;
            if (up > best)   { best = up;   dir = 1; }
            if (left > best) { best = left; dir = 2; }
            H[ii*Hn+jj] = best;
            trace[ii*Hn+jj] = dir;
        }
    }
}

/* ── NW traceback → aligned pair indices ──────────────────────── */

inline int nw_traceback(const std::vector<int8_t> &trace, int Np, int Nn,
                        int *align_p, int *align_n, int max_pairs) {
    int Hn = Nn+1, n_aligned = 0;
    int i = Np, j = Nn;
    int tmp_p[4096], tmp_n[4096];
    int count = 0;
    while (i > 0 && j > 0 && count < 4096) {
        int8_t dir = trace[i*Hn+j];
        if (dir == 0) {
            tmp_p[count] = i-1;
            tmp_n[count] = j-1;
            count++;
            i--; j--;
        } else if (dir == 1) {
            i--;
        } else {
            j--;
        }
    }
    n_aligned = std::min(count, max_pairs);
    for (int k = 0; k < n_aligned; k++) {
        align_p[k] = tmp_p[count-1-k];
        align_n[k] = tmp_n[count-1-k];
    }
    return n_aligned;
}

#endif
