#ifndef INDIALIGN_SEARCH_H
#define INDIALIGN_SEARCH_H

#include "core.h"
#include "seeds.h"
#include <cstring>
#include <cmath>

/* ── Iterative seed refinement (20 fixed iterations) ──────────── */

inline void iterative_seed_refine(const float *pred, const float *native,
                                  const uint8_t *valid,
                                  const uint8_t *seed_mask, float d0_search,
                                  int N, int max_iter,
                                  float *R, float *t) {
    uint8_t work[4096];
    for (int i = 0; i < N; i++) work[i] = seed_mask[i] & valid[i];
    kabsch(pred, native, work, N, R, t);
    float d0s_sq = d0_search * d0_search;
    for (int iter = 0; iter < max_iter; iter++) {
        uint8_t sel[4096];
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            if (!valid[i]) { sel[i] = 0; continue; }
            float mx = pred[i*3]*R[0]+pred[i*3+1]*R[3]+pred[i*3+2]*R[6]+t[0];
            float my = pred[i*3]*R[1]+pred[i*3+1]*R[4]+pred[i*3+2]*R[7]+t[1];
            float mz = pred[i*3]*R[2]+pred[i*3+1]*R[5]+pred[i*3+2]*R[8]+t[2];
            float dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
            float d2 = dx*dx+dy*dy+dz*dz;
            sel[i] = (d2 <= d0s_sq) ? 1 : 0;
            cnt += sel[i];
        }
        if (cnt >= 3)
            kabsch(pred, native, sel, N, R, t);
    }
}

/* ── Evaluate seed bank: K seeds × C d0 candidates ───────────── */

struct SeedResult { float R[9], t[3], score; };

inline SeedResult evaluate_seed_bank(
    const float *pred, const float *native, const uint8_t *valid,
    const uint8_t *seeds, int K,
    const float *d0_candidates, int C,
    float d0, float score_d8, float Lnorm, int N, int max_iter,
    bool use_score_d8 = true)
{
    SeedResult best;
    best.score = -1e30f;

    #pragma omp parallel
    {
        SeedResult local_best;
        local_best.score = -1e30f;
        float R[9], t[3];

        #pragma omp for collapse(2) schedule(dynamic)
        for (int k = 0; k < K; k++) {
            for (int c = 0; c < C; c++) {
                const uint8_t *seed = seeds + k * N;
                iterative_seed_refine(pred, native, valid, seed,
                                      d0_candidates[c], N, max_iter, R, t);
                float sc;
                if (use_score_d8)
                    sc = tm_score(pred, native, valid, R, t, d0, Lnorm, score_d8, N);
                else
                    sc = tm_score_no_d8(pred, native, valid, R, t, d0, Lnorm, N);
                if (sc > local_best.score) {
                    local_best.score = sc;
                    std::memcpy(local_best.R, R, 36);
                    std::memcpy(local_best.t, t, 12);
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
