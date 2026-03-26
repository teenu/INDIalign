#include "api.h"
#include "core.h"
#include "seeds.h"
#include "search.h"
#include "align.h"
#include "rescue.h"
#include "threading.h"
#include "ss.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef INDIALIGN_CUDA
extern SeedResult gpu_evaluate_seed_bank(
    const double *pred, const double *native, const uint8_t *valid,
    const uint8_t *seeds, int K,
    const double *d0_cands, int C,
    double d0, double score_d8, double Lnorm, int N, int max_iter);
extern bool gpu_available();
#else
static inline bool gpu_available() { return false; }
#endif

static SeedResult do_tmscore_search(
    const double *pred, const double *native,
    const uint8_t *valid, const uint8_t *pv, const uint8_t *nv,
    int N, double Lnorm, int max_iter, bool use_frag, int dp_iter,
    [[maybe_unused]] bool use_cuda)
{
    const bool profile = std::getenv("INDIALIGN_PROFILE") != nullptr;
    const char *profile_tag = std::getenv("INDIALIGN_PROFILE_TAG");
    const auto profile_stage = [&](const char *stage,
                                   const std::chrono::steady_clock::time_point &start) {
        if (!profile) return;
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count();
        std::fprintf(stderr, "[indialign]%s N=%d stage=%s ms=%.3f\n",
                     profile_tag ? profile_tag : "", N, stage, ms);
    };

    if (Lnorm <= 2.0) {
        SeedResult r; r.score = 0;
        double I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(r.R,I,72);
        r.t[0]=r.t[1]=r.t[2]=0; return r;
    }
    double d0, d0s, sd8;
    d0_from_length_rna(Lnorm, &d0, &d0s, &sd8);
    // d0 search candidates: d0s × multipliers + d0 + add_offsets
    static constexpr double D0_ADD_OFFSETS[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    static constexpr int N_ADD = 5;
    double d0c_all[N_D0_MULTS + N_ADD];
    int C_total = 0;
    for (int c = 0; c < N_D0_MULTS; c++) d0c_all[C_total++] = d0s * D0_MULTS[c];
    for (int c = 0; c < N_ADD; c++) {
        double v = d0 + D0_ADD_OFFSETS[c];
        // Add only if not a duplicate
        bool dup = false;
        for (int e = 0; e < C_total && !dup; e++)
            dup = (std::abs(d0c_all[e] - v) < 0.01);
        if (!dup) d0c_all[C_total++] = v;
    }

    // 1. Build fragment seeds + anchor contact seeds
    std::vector<uint8_t> smasks;
    int K = 0;
    auto stage_start = std::chrono::steady_clock::now();
    build_seed_masks(N, valid, use_frag, smasks, K);
    build_anchor_contact_seeds(pred, native, valid, N, d0s, d0, 5,
                               smasks, K);
    profile_stage("seed_build", stage_start);
    if (K == 0) {
        SeedResult r; r.score = 0;
        double I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(r.R,I,72);
        r.t[0]=r.t[1]=r.t[2]=0; return r;
    }

    // Rescore a candidate R,t with same-index tm_score_no_d8.
    // Cross-index DP functions return scores on compact aligned pairs;
    // this converts to the actual same-index TM-score on full arrays.
    auto rescore = [&](SeedResult &r) {
        r.score = tm_score_no_d8(pred, native, valid, r.R, r.t, d0, Lnorm, N);
    };

    // 2. Evaluate seed bank (same-index, no d8 filter)
    SeedResult best;
    stage_start = std::chrono::steady_clock::now();
#ifdef INDIALIGN_CUDA
    if (use_cuda && gpu_available())
        best = gpu_evaluate_seed_bank(pred, native, valid,
                                      smasks.data(), K, d0c_all, C_total,
                                      d0, sd8, Lnorm, N, max_iter);
    else
#endif
        best = evaluate_seed_bank(pred, native, valid,
                                  smasks.data(), K, d0c_all, C_total,
                                  d0, sd8, Lnorm, N, max_iter, false);
    profile_stage("seed_bank", stage_start);

    // 2b. SS-based alignment seeds (IA2 + IA4) — cross-index, rescore
    char ss_p[4096], ss_n[4096];
    stage_start = std::chrono::steady_clock::now();
    assign_rna_ss(pred, pv, N, ss_p);
    assign_rna_ss(native, nv, N, ss_n);
    auto ss = evaluate_ss_seeds(pred, native, valid, pv, nv,
                                ss_p, ss_n, best.R, best.t,
                                d0, d0s, sd8, Lnorm, N, dp_iter);
    rescore(ss);
    if (ss.score > best.score) best = ss;
    profile_stage("ss", stage_start);

    // 2c. Gapless threading (same-index scoring)
    stage_start = std::chrono::steady_clock::now();
    auto gt = gapless_threading_search(pred, native, pv, nv, valid,
                                       d0, d0s, Lnorm, N);
    if (gt.score > best.score) best = gt;
    profile_stage("gapless_threading", stage_start);

    // 3. Rescue strategies — all cross-index, rescore each
    bool has_cross = false;
    for (int i = 0; i < N && !has_cross; i++)
        has_cross = (pv[i] != nv[i]);
    if (best.score < 0.5) {
        stage_start = std::chrono::steady_clock::now();
        auto lf = evaluate_local_fragment_dp(pred, native, valid, pv, nv,
                                              d0, d0s, Lnorm, N, false);
        rescore(lf);
        if (lf.score > best.score) best = lf;
        profile_stage("local_fragment", stage_start);
    }
    if (best.score < 0.5) {
        stage_start = std::chrono::steady_clock::now();
        auto th = evaluate_threading_dp(pred, native, valid, pv, nv,
                                         d0, d0s, Lnorm, N);
        rescore(th);
        if (th.score > best.score) best = th;
        profile_stage("threading_dp", stage_start);
    }
    if (best.score < 0.3) {
        stage_start = std::chrono::steady_clock::now();
        auto dl = evaluate_local_fragment_dp(pred, native, valid, pv, nv,
                                              d0, d0s, Lnorm, N, true);
        rescore(dl);
        if (dl.score > best.score) best = dl;
        profile_stage("dense_local_fragment", stage_start);
    }

    // 4. Two-pass distance seed (same-index scoring)
    stage_start = std::chrono::steady_clock::now();
    std::vector<double> d2v(N);
    dist2_fused(pred, native, valid, best.R, best.t, d2v.data(), N);
    std::vector<uint8_t> topk_mask(N);
    build_topk_seed(d2v.data(), valid, N, 0.5, topk_mask.data());
    int topk_cnt = 0;
    for (int i = 0; i < N; i++) topk_cnt += topk_mask[i];
    if (topk_cnt >= 3) {
        SeedResult tp;
#ifdef INDIALIGN_CUDA
        if (use_cuda && gpu_available())
            tp = gpu_evaluate_seed_bank(pred, native, valid,
                                        topk_mask.data(), 1, d0c_all, C_total,
                                        d0, sd8, Lnorm, N, max_iter);
        else
#endif
            tp = evaluate_seed_bank(pred, native, valid,
                                    topk_mask.data(), 1, d0c_all, C_total,
                                    d0, sd8, Lnorm, N, max_iter, false);
        if (tp.score > best.score) best = tp;
    }
    profile_stage("topk_seed", stage_start);

    // 5. DP refinement — all candidates rescored with same-index
    auto update_best = [&](const double *R, const double *t, double sc) {
        if (sc > best.score) {
            best.score = sc;
            if (R != best.R) std::memcpy(best.R, R, 72);
            if (t != best.t) std::memcpy(best.t, t, 24);
        }
    };

    if (dp_iter > 0) {
        stage_start = std::chrono::steady_clock::now();
        double seed_score = best.score;
        auto dp = dp_refine(pred, native, valid, pv, nv,
                            best.R, best.t, d0, d0s, sd8, Lnorm,
                            N, dp_iter, false);
        double dp_id = tm_score_no_d8(pred, native, valid,
                                      dp.R, dp.t, d0, Lnorm, N);
        update_best(dp.R, dp.t, dp_id);
        double orig_id = tm_score_no_d8(pred, native, valid,
                                        best.R, best.t, d0, Lnorm, N);
        update_best(best.R, best.t, orig_id);

        if (has_cross) {
            double I[9]={1,0,0,0,1,0,0,0,1}, z[3]={};
            auto idp = dp_refine(pred, native, valid, pv, nv,
                                 I, z, d0, d0s, sd8, Lnorm,
                                 N, dp_iter, true);
            double idp_id = tm_score_no_d8(pred, native, valid,
                                           idp.R, idp.t, d0, Lnorm, N);
            update_best(idp.R, idp.t, idp_id);

            int extra = std::max(0, 5 - dp_iter);
            if (extra > 0 && best.score < 0.36) {
                bool use_seed = (seed_score > dp_id);
                const double *eR = use_seed ? best.R : dp.R;
                const double *et = use_seed ? best.t : dp.t;
                auto ex = dp_refine(pred, native, valid, pv, nv,
                                    eR, et, d0, d0s, sd8, Lnorm,
                                    N, extra, true);
                double ex_id = tm_score_no_d8(pred, native, valid,
                                              ex.R, ex.t, d0, Lnorm, N);
                update_best(ex.R, ex.t, ex_id);
            }
        }
        profile_stage("dp_refine", stage_start);
    } else {
        double id_sc = tm_score_no_d8(pred, native, valid,
                                      best.R, best.t, d0, Lnorm, N);
        update_best(best.R, best.t, id_sc);
    }
    // 5b. DP from raw cross-offset Kabsch starting points
    if (dp_iter > 0) {
        stage_start = std::chrono::steady_clock::now();
        double Pa[4096*3], Qa[4096*3];
        uint8_t pm[4096];
        int off_stride = std::max(1, N / 10);
        for (int k = -N/2; k <= N/2; k += off_stride) {
            if (k == 0) continue;
            int na = 0;
            for (int i = 0; i < N; i++) {
                int j = i - k;
                if (j < 0 || j >= N || !pv[i] || !nv[j]) continue;
                std::memcpy(&Pa[na*3], &pred[i*3], 24);
                std::memcpy(&Qa[na*3], &native[j*3], 24);
                pm[na++] = 1;
            }
            if (na < 3) continue;
            double sR[9], st[3];
            kabsch(Pa, Qa, pm, na, sR, st);
            auto tdp = dp_refine(pred, native, valid, pv, nv,
                                 sR, st, d0, d0s, sd8, Lnorm,
                                 N, dp_iter, true);
            double sc = tm_score_no_d8(pred, native, valid,
                                       tdp.R, tdp.t, d0, Lnorm, N);
            update_best(tdp.R, tdp.t, sc);
        }
        profile_stage("offset_dp", stage_start);
    }

    // 6. Weighted TM refinement (4 iterations)
    stage_start = std::chrono::steady_clock::now();
    double wR[9], wt[3], wsc;
    std::memcpy(wR, best.R, 72); std::memcpy(wt, best.t, 24);
    weighted_tm_refine(pred, native, valid, d0, Lnorm, N, 4, wR, wt, &wsc);
    update_best(wR, wt, wsc);
    profile_stage("weighted_refine", stage_start);
    return best;
}

/* ── C API ────────────────────────────────────────────────────── */

extern "C" {

void indialign_default_config(NativeConfig *cfg) {
    cfg->backend_mode = 0;
    cfg->use_fragment_search = 1;
    cfg->max_iter = 20;
    cfg->dp_iter = 1;
    cfg->max_mem_gb = 20.0;
    cfg->request_cuda = 0;
}

int indialign_tmscore_search(const NativePairInput *inp,
                             const NativeConfig *cfg,
                             NativePairResult *out) {
    int N = inp->length;
    if (N <= 0 || N > 4096) return -1;

    auto res = do_tmscore_search(
        inp->pred_xyz, inp->native_xyz,
        inp->valid_mask, inp->pred_valid_mask, inp->native_valid_mask,
        N, inp->lnorm,
        cfg->max_iter, cfg->use_fragment_search != 0,
        cfg->dp_iter, cfg->request_cuda != 0);

    out->score = res.score;
    for (int i = 0; i < 9; i++) out->rotation[i] = res.R[i];
    for (int i = 0; i < 3; i++) out->translation[i] = res.t[i];
    return 0;
}

int indialign_tmscore_search_batch(int n_pairs,
                                    const NativePairInput *inputs,
                                    const NativeConfig *cfg,
                                    NativePairResult *outputs) {
    #pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < n_pairs; p++)
        indialign_tmscore_search(&inputs[p], cfg, &outputs[p]);
    return 0;
}

int indialign_cuda_available(void) {
#ifdef INDIALIGN_CUDA
    return gpu_available() ? 1 : 0;
#else
    return 0;
#endif
}

} // extern "C"
