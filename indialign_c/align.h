#ifndef INDIALIGN_ALIGN_H
#define INDIALIGN_ALIGN_H

#include "core.h"
#include "nw.h"
#include "seeds.h"
#include "search.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>

/* ── Detailed search on aligned pairs ─────────────────────────── */

inline SeedResult alignment_detailed_search(
    const float *pred, const float *native,
    const int *align_p, const int *align_n, int n_aligned,
    float d0, float d0_search, float score_d8, float Lnorm,
    int /*N*/, bool use_frag, int max_iter_override = -1,
    int max_frag_starts = MAX_FRAG_STARTS)
{
    if (n_aligned < 3) {
        SeedResult r; r.score = -1e30f;
        float I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(r.R,I,36);
        r.t[0]=r.t[1]=r.t[2]=0; return r;
    }
    // Extract aligned coordinates
    std::vector<float> P(n_aligned*3), Q(n_aligned*3);
    for (int i = 0; i < n_aligned; i++) {
        std::memcpy(&P[i*3], &pred[align_p[i]*3], 12);
        std::memcpy(&Q[i*3], &native[align_n[i]*3], 12);
    }
    std::vector<uint8_t> pmask(n_aligned, 1);
    // Build seeds over aligned pairs
    std::vector<uint8_t> smasks;
    int K = 0;
    build_seed_masks(n_aligned, pmask.data(), use_frag, smasks, K, max_frag_starts);
    if (K == 0) {
        SeedResult r; r.score = -1e30f;
        float I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(r.R,I,36);
        r.t[0]=r.t[1]=r.t[2]=0; return r;
    }
    float d0c[N_D0_MULTS];
    for (int c = 0; c < N_D0_MULTS; c++) d0c[c] = d0_search * D0_MULTS[c];
    int mi = (max_iter_override >= 0) ? max_iter_override :
             (use_frag ? 20 : std::min(20, 8));
    return evaluate_seed_bank(P.data(), Q.data(), pmask.data(),
                              smasks.data(), K, d0c, N_D0_MULTS,
                              d0, score_d8, Lnorm, n_aligned, mi);
}

/* ── DP refinement chunk ──────────────────────────────────────── */

inline SeedResult dp_refine_chunk(
    const float *pred, const float *native, const uint8_t * /*valid*/,
    const uint8_t *pv, const uint8_t *nv,
    const float *R0, const float *t0,
    float d0, float d0_search, float score_d8, float Lnorm,
    int N, int max_iter, bool d0_search_lite)
{
    SeedResult best;
    best.score = 0;
    float I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(best.R,I,36);
    best.t[0]=best.t[1]=best.t[2]=0;
    float curR[9], curt[3];
    std::memcpy(curR, R0, 36); std::memcpy(curt, t0, 12);

    std::vector<float> moved(N*3), smat(N*N), H_nw;
    std::vector<int8_t> trace_nw;
    int ap[4096], an[4096];

    for (int it = 0; it < max_iter; it++) {
        apply_transform(pred, curR, curt, moved.data(), N);
        score_matrix(moved.data(), native, pv, nv, d0, score_d8, N, smat.data());
        nw_dp(smat.data(), N, N, H_nw, trace_nw);
        int na = nw_traceback(trace_nw, N, N, ap, an, std::min(N, 4096));
        if (na < 3) continue;
        // Build pair mask and extract aligned coords
        std::vector<uint8_t> pmask(na);
        std::vector<float> Pa(na*3), Qa(na*3);
        int valid_cnt = 0;
        for (int i = 0; i < na; i++) {
            pmask[i] = pv[ap[i]] && nv[an[i]];
            std::memcpy(&Pa[i*3], &pred[ap[i]*3], 12);
            std::memcpy(&Qa[i*3], &native[an[i]*3], 12);
            valid_cnt += pmask[i];
        }
        if (valid_cnt < 3) continue;
        float newR[9], newt[3];
        kabsch(Pa.data(), Qa.data(), pmask.data(), na, newR, newt);
        // Score direct Kabsch on aligned pairs
        float d0sq = std::max(d0*d0, 1e-12f);
        float it_score = 0;
        for (int i = 0; i < na; i++) {
            if (!pmask[i]) continue;
            float mx = Pa[i*3]*newR[0]+Pa[i*3+1]*newR[3]+Pa[i*3+2]*newR[6]+newt[0];
            float my = Pa[i*3]*newR[1]+Pa[i*3+1]*newR[4]+Pa[i*3+2]*newR[7]+newt[1];
            float mz = Pa[i*3]*newR[2]+Pa[i*3+1]*newR[5]+Pa[i*3+2]*newR[8]+newt[2];
            float dx=mx-Qa[i*3], dy=my-Qa[i*3+1], dz=mz-Qa[i*3+2];
            float d2 = dx*dx+dy*dy+dz*dz;
            it_score += 1.0f / (1.0f + d2/d0sq);
        }
        it_score /= std::max(Lnorm, 1.0f);
        float itR[9], itt[3];
        std::memcpy(itR, newR, 36); std::memcpy(itt, newt, 12);
        // Fragment search refinement on aligned pairs
        bool use_frag = !d0_search_lite;
        auto det = alignment_detailed_search(
            pred, native, ap, an, na,
            d0, d0_search, score_d8, Lnorm, N, use_frag);
        if (det.score > it_score) {
            std::memcpy(itR, det.R, 36);
            std::memcpy(itt, det.t, 12);
            it_score = det.score;
        }
        std::memcpy(curR, itR, 36); std::memcpy(curt, itt, 12);
        if (it_score > best.score) {
            best.score = it_score;
            std::memcpy(best.R, itR, 36);
            std::memcpy(best.t, itt, 12);
        }
    }
    return best;
}

/* ── dp_refine wrapper ────────────────────────────────────────── */

inline SeedResult dp_refine(
    const float *pred, const float *native, const uint8_t *valid,
    const uint8_t *pv, const uint8_t *nv,
    const float *R0, const float *t0,
    float d0, float d0_search, float score_d8, float Lnorm,
    int N, int max_iter, bool d0_search_lite = false)
{
    return dp_refine_chunk(pred, native, valid, pv, nv,
                           R0, t0, d0, d0_search, score_d8, Lnorm,
                           N, max_iter, d0_search_lite);
}

#endif
