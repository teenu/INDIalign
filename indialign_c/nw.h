#ifndef INDIALIGN_NW_H
#define INDIALIGN_NW_H

#include <cstring>
#include <algorithm>
#include <vector>

/* ── Needleman-Wunsch DP with optional gap-open penalty ──────── */

inline void nw_dp(const double *smat, int Np, int Nn,
                  std::vector<double> &H, std::vector<int8_t> &trace,
                  double gap_open = 0.0) {
    int Hp = Np+1, Hn = Nn+1;
    H.assign(Hp * Hn, 0.0);
    trace.assign(Hp * Hn, -1); // -1 = boundary/gap (not from diagonal)
    for (int d = 2; d < Np + Nn + 2; d++) {
        int ilo = std::max(1, d - Nn);
        int ihi = std::min(Np, d - 1);
        for (int ii = ilo; ii <= ihi; ii++) {
            int jj = d - ii;
            double diag = H[(ii-1)*Hn+(jj-1)] + smat[(ii-1)*Nn+(jj-1)];
            double up   = H[(ii-1)*Hn+jj];
            if (trace[(ii-1)*Hn+jj] == 0) up += gap_open;
            double left = H[ii*Hn+(jj-1)];
            if (trace[ii*Hn+(jj-1)] == 0) left += gap_open;
            if (diag >= up && diag >= left) {
                H[ii*Hn+jj] = diag;
                trace[ii*Hn+jj] = 0;
            } else if (left >= up) {
                H[ii*Hn+jj] = left;
                trace[ii*Hn+jj] = 2;
            } else {
                H[ii*Hn+jj] = up;
                trace[ii*Hn+jj] = 1;
            }
        }
    }
}

/* ── NW traceback → aligned pair indices ──────────────────────── */

inline int nw_traceback(const std::vector<int8_t> &trace, int Np, int Nn,
                        int *align_p, int *align_n, int max_pairs) {
    int Hn = Nn+1;
    int i = Np, j = Nn;
    std::vector<int> tmp_p, tmp_n;
    tmp_p.reserve(std::min(Np, Nn));
    tmp_n.reserve(std::min(Np, Nn));
    while (i > 0 && j > 0) {
        int8_t dir = trace[i*Hn+j];
        if (dir == 0) {
            tmp_p.push_back(i-1);
            tmp_n.push_back(j-1);
            i--; j--;
        } else if (dir == 1) {
            i--;
        } else {
            j--;
        }
    }
    int count = (int)tmp_p.size();
    int n_aligned = std::min(count, max_pairs);
    for (int k = 0; k < n_aligned; k++) {
        align_p[k] = tmp_p[count-1-k];
        align_n[k] = tmp_n[count-1-k];
    }
    return n_aligned;
}

#endif
