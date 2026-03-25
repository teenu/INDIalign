#ifndef INDIALIGN_SEARCH_H
#define INDIALIGN_SEARCH_H

#include "core.h"
#include "seeds.h"
#include <cstring>
#include <cmath>

/* ── Iterative seed refinement with convergence detection ─────── */

inline void iterative_seed_refine(const double *pred, const double *native,
                                  const uint8_t *valid,
                                  const uint8_t *seed_mask, double d0_search,
                                  int N, int max_iter,
                                  double *R, double *t) {
    uint8_t work[4096];
    for (int i = 0; i < N; i++) work[i] = seed_mask[i] & valid[i];
    kabsch(pred, native, work, N, R, t);
    double d0s_sq = d0_search * d0_search;
    int prev_cnt = -1;
    for (int iter = 0; iter < max_iter; iter++) {
        uint8_t sel[4096];
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            if (!valid[i]) { sel[i] = 0; continue; }
            double mx = pred[i*3]*R[0]+pred[i*3+1]*R[3]+pred[i*3+2]*R[6]+t[0];
            double my = pred[i*3]*R[1]+pred[i*3+1]*R[4]+pred[i*3+2]*R[7]+t[1];
            double mz = pred[i*3]*R[2]+pred[i*3+1]*R[5]+pred[i*3+2]*R[8]+t[2];
            double dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
            double d2 = dx*dx+dy*dy+dz*dz;
            sel[i] = (d2 <= d0s_sq) ? 1 : 0;
            cnt += sel[i];
        }
        if (cnt >= 3)
            kabsch(pred, native, sel, N, R, t);
        // Convergence: stop if pair count stabilized (same count implies
        // same set since distances change monotonically with R,t updates)
        if (cnt == prev_cnt) break;
        prev_cnt = cnt;
    }
}

/* ── Evaluate seed bank: K seeds × C d0 candidates ───────────── */

struct SeedResult { double R[9], t[3], score; };

inline SeedResult evaluate_seed_bank(
    const double *pred, const double *native, const uint8_t *valid,
    const uint8_t *seeds, int K,
    const double *d0_candidates, int C,
    double d0, double score_d8, double Lnorm, int N, int max_iter,
    bool use_score_d8 = true)
{
    SeedResult best;
    best.score = -1e30;

    #pragma omp parallel
    {
        SeedResult local_best;
        local_best.score = -1e30;
        double R[9], t[3];

        #pragma omp for collapse(2) schedule(dynamic)
        for (int k = 0; k < K; k++) {
            for (int c = 0; c < C; c++) {
                const uint8_t *seed = seeds + k * N;
                iterative_seed_refine(pred, native, valid, seed,
                                      d0_candidates[c], N, max_iter, R, t);
                double sc;
                if (use_score_d8)
                    sc = tm_score(pred, native, valid, R, t, d0, Lnorm, score_d8, N);
                else
                    sc = tm_score_no_d8(pred, native, valid, R, t, d0, Lnorm, N);
                if (sc > local_best.score) {
                    local_best.score = sc;
                    std::memcpy(local_best.R, R, 72);
                    std::memcpy(local_best.t, t, 24);
                }
            }
        }

        #pragma omp critical
        {
            if (local_best.score > best.score) best = local_best;
        }
    }
    return best;
}

#endif
