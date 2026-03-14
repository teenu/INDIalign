#include "api.h"
#include "core.h"
#include "seeds.h"
#include "search.h"
#include "align.h"
#include "rescue.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef INDIALIGN_CUDA
extern SeedResult gpu_evaluate_seed_bank(
    const float *pred, const float *native, const uint8_t *valid,
    const uint8_t *seeds, int K,
    const float *d0_cands, int C,
    float d0, float score_d8, float Lnorm, int N, int max_iter);
extern bool gpu_available();
#else
static inline bool gpu_available() { return false; }
#endif

static SeedResult do_tmscore_search(
    const float *pred, const float *native,
    const uint8_t *valid, const uint8_t *pv, const uint8_t *nv,
    int N, float Lnorm, int max_iter, bool use_frag, int dp_iter,
    [[maybe_unused]] bool use_cuda)
{
    if (Lnorm <= 2.0f) {
        SeedResult r; r.score = 0;
        float I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(r.R,I,36);
        r.t[0]=r.t[1]=r.t[2]=0; return r;
    }
    float d0, d0s, sd8;
    d0_from_length_rna(Lnorm, &d0, &d0s, &sd8);
    // d0 search candidates: d0s × multipliers + d0 + add_offsets
    static constexpr float D0_ADD_OFFSETS[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    static constexpr int N_ADD = 5;
    float d0c_all[N_D0_MULTS + N_ADD];
    int C_total = 0;
    for (int c = 0; c < N_D0_MULTS; c++) d0c_all[C_total++] = d0s * D0_MULTS[c];
    for (int c = 0; c < N_ADD; c++) {
        float v = d0 + D0_ADD_OFFSETS[c];
        // Add only if not a duplicate
        bool dup = false;
        for (int e = 0; e < C_total && !dup; e++)
            dup = (std::abs(d0c_all[e] - v) < 0.01f);
        if (!dup) d0c_all[C_total++] = v;
    }

    // 1. Build fragment seeds
    std::vector<uint8_t> smasks;
    int K = 0;
    build_seed_masks(N, valid, use_frag, smasks, K);
    if (K == 0) {
        SeedResult r; r.score = 0;
        float I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(r.R,I,36);
        r.t[0]=r.t[1]=r.t[2]=0; return r;
    }

    // 2. Evaluate seed bank (no d8 filter for scoring, matching weighted_all)
    SeedResult best;
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
    // 3. Cross-index rescue (when structures differ in length)
    bool has_cross = false;
    for (int i = 0; i < N && !has_cross; i++)
        has_cross = (pv[i] != nv[i]);
    if (has_cross && best.score < 0.5f) {
        auto lf = evaluate_local_fragment_dp(pred, native, valid, pv, nv,
                                              d0, d0s, Lnorm, N, false);
        if (lf.score > best.score) best = lf;
    }
    if (has_cross && best.score < 0.5f) {
        auto th = evaluate_threading_dp(pred, native, valid, pv, nv,
                                         d0, d0s, Lnorm, N);
        if (th.score > best.score) best = th;
    }
    if (has_cross && best.score < 0.3f) {
        auto dl = evaluate_local_fragment_dp(pred, native, valid, pv, nv,
                                              d0, d0s, Lnorm, N, true);
        if (dl.score > best.score) best = dl;
    }

    // 4. Two-pass distance seed
    std::vector<float> d2v(N);
    dist2_fused(pred, native, valid, best.R, best.t, d2v.data(), N);
    std::vector<uint8_t> topk_mask(N);
    build_topk_seed(d2v.data(), valid, N, 0.5f, topk_mask.data());
    int topk_cnt = 0;
    for (int i = 0; i < N; i++) topk_cnt += topk_mask[i];
    if (topk_cnt >= 3) {
        auto tp = evaluate_seed_bank(pred, native, valid,
                                     topk_mask.data(), 1, d0c_all, C_total,
                                     d0, sd8, Lnorm, N, max_iter, false);
        if (tp.score > best.score) best = tp;
    }

    // 5. DP refinement — track best R,t across all candidates
    auto update_best = [&](const float *R, const float *t, float sc) {
        if (sc > best.score) {
            best.score = sc;
            if (R != best.R) std::memcpy(best.R, R, 36);
            if (t != best.t) std::memcpy(best.t, t, 12);
        }
    };

    if (dp_iter > 0) {
        float seed_score = best.score;
        auto dp = dp_refine(pred, native, valid, pv, nv,
                            best.R, best.t, d0, d0s, sd8, Lnorm,
                            N, dp_iter, false);
        update_best(dp.R, dp.t, dp.score);
        float dp_id = tm_score_no_d8(pred, native, valid,
                                      dp.R, dp.t, d0, Lnorm, N);
        update_best(dp.R, dp.t, dp_id);
        float orig_id = tm_score_no_d8(pred, native, valid,
                                        best.R, best.t, d0, Lnorm, N);
        update_best(best.R, best.t, orig_id);

        if (has_cross) {
            float I[9]={1,0,0,0,1,0,0,0,1}, z[3]={};
            auto idp = dp_refine(pred, native, valid, pv, nv,
                                 I, z, d0, d0s, sd8, Lnorm,
                                 N, dp_iter, true);
            update_best(idp.R, idp.t, idp.score);
            float idp_id = tm_score_no_d8(pred, native, valid,
                                           idp.R, idp.t, d0, Lnorm, N);
            update_best(idp.R, idp.t, idp_id);

            int extra = std::max(0, 5 - dp_iter);
            if (extra > 0 && best.score < 0.36f) {
                bool use_seed = (seed_score > dp.score);
                const float *eR = use_seed ? best.R : dp.R;
                const float *et = use_seed ? best.t : dp.t;
                auto ex = dp_refine(pred, native, valid, pv, nv,
                                    eR, et, d0, d0s, sd8, Lnorm,
                                    N, extra, true);
                update_best(ex.R, ex.t, ex.score);
                float ex_id = tm_score_no_d8(pred, native, valid,
                                              ex.R, ex.t, d0, Lnorm, N);
                update_best(ex.R, ex.t, ex_id);
            }
        }
    } else {
        float id_sc = tm_score_no_d8(pred, native, valid,
                                      best.R, best.t, d0, Lnorm, N);
        update_best(best.R, best.t, id_sc);
    }

    // 6. Weighted TM refinement (4 iterations)
    float wR[9], wt[3], wsc;
    std::memcpy(wR, best.R, 36); std::memcpy(wt, best.t, 12);
    weighted_tm_refine(pred, native, valid, d0, Lnorm, N, 4, wR, wt, &wsc);
    update_best(wR, wt, wsc);
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

    std::vector<float> pred(N*3), native(N*3);
    for (int i = 0; i < N*3; i++) {
        pred[i]   = (float)inp->pred_xyz[i];
        native[i] = (float)inp->native_xyz[i];
    }
    auto res = do_tmscore_search(
        pred.data(), native.data(),
        inp->valid_mask, inp->pred_valid_mask, inp->native_valid_mask,
        N, (float)inp->lnorm,
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
