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
    const double *pred, const double *native,
    const int *align_p, const int *align_n, int n_aligned,
    double d0, double d0_search, double score_d8, double Lnorm,
    int /*N*/, bool use_frag, int max_iter_override = -1,
    int max_frag_starts = MAX_FRAG_STARTS)
{
    if (n_aligned < 3) {
        SeedResult r; r.score = -1e30;
        double I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(r.R,I,72);
        r.t[0]=r.t[1]=r.t[2]=0; return r;
    }
    // Extract aligned coordinates
    std::vector<double> P(n_aligned*3), Q(n_aligned*3);
    for (int i = 0; i < n_aligned; i++) {
        std::memcpy(&P[i*3], &pred[align_p[i]*3], 24);
        std::memcpy(&Q[i*3], &native[align_n[i]*3], 24);
    }
    std::vector<uint8_t> pmask(n_aligned, 1);
    // Build seeds over aligned pairs
    std::vector<uint8_t> smasks;
    int K = 0;
    build_seed_masks(n_aligned, pmask.data(), use_frag, smasks, K, max_frag_starts);
    if (K == 0) {
        SeedResult r; r.score = -1e30;
        double I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(r.R,I,72);
        r.t[0]=r.t[1]=r.t[2]=0; return r;
    }
    double d0c[N_D0_MULTS];
    for (int c = 0; c < N_D0_MULTS; c++) d0c[c] = d0_search * D0_MULTS[c];
    int mi = (max_iter_override >= 0) ? max_iter_override :
             (use_frag ? 20 : std::min(20, 8));
    return evaluate_seed_bank(P.data(), Q.data(), pmask.data(),
                              smasks.data(), K, d0c, N_D0_MULTS,
                              d0, score_d8, Lnorm, n_aligned, mi);
}

/* ── DP refinement chunk (compact NW on valid residues only) ── */

inline SeedResult dp_refine_chunk(
    const double *pred, const double *native, const uint8_t * /*valid*/,
    const uint8_t *pv, const uint8_t *nv,
    const double *R0, const double *t0,
    double d0, double d0_search, double score_d8, double Lnorm,
    int N, int max_iter, bool d0_search_lite)
{
    SeedResult best;
    best.score = 0;
    double I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(best.R,I,72);
    best.t[0]=best.t[1]=best.t[2]=0;

    // Map valid indices for compact DP (plen×nlen instead of N×N)
    int pv_map[4096], nv_map[4096];
    int plen = 0, nlen = 0;
    for (int i = 0; i < N; i++) if (pv[i]) pv_map[plen++] = i;
    for (int i = 0; i < N; i++) if (nv[i]) nv_map[nlen++] = i;
    if (plen < 3 || nlen < 3) return best;

    std::vector<double> moved(N*3), smat(plen * nlen), H_nw;
    std::vector<int8_t> trace_nw;
    int ap[4096], an[4096];
    double d0sq_s = std::max(d0*d0, 1e-12), sd8sq = score_d8*score_d8;

    static constexpr double gap_opens[] = {-0.6, 0.0};
    for (int g = 0; g < 2; g++) {
        double curR[9], curt[3];
        std::memcpy(curR, R0, 72); std::memcpy(curt, t0, 24);

        for (int it = 0; it < max_iter; it++) {
            apply_transform(pred, curR, curt, moved.data(), N);
            for (int pi = 0; pi < plen; pi++) {
                int i = pv_map[pi];
                for (int ni = 0; ni < nlen; ni++) {
                    int j = nv_map[ni];
                    double dx = moved[i*3]-native[j*3];
                    double dy = moved[i*3+1]-native[j*3+1];
                    double dz = moved[i*3+2]-native[j*3+2];
                    double d2 = dx*dx+dy*dy+dz*dz;
                    smat[pi*nlen+ni] = (d2 <= sd8sq) ?
                        1.0/(1.0 + d2/d0sq_s) : 0.0;
                }
            }
            nw_dp(smat.data(), plen, nlen, H_nw, trace_nw, gap_opens[g]);
            int na = nw_traceback(trace_nw, plen, nlen, ap, an,
                                  std::min(std::min(plen,nlen), 4096));
            for (int i = 0; i < na; i++) {
                ap[i] = pv_map[ap[i]];
                an[i] = nv_map[an[i]];
            }
            if (na < 3) continue;
            std::vector<double> Pa(na*3), Qa(na*3);
            std::vector<uint8_t> pmask(na, 1);
            for (int i = 0; i < na; i++) {
                std::memcpy(&Pa[i*3], &pred[ap[i]*3], 24);
                std::memcpy(&Qa[i*3], &native[an[i]*3], 24);
            }
            double newR[9], newt[3];
            kabsch(Pa.data(), Qa.data(), pmask.data(), na, newR, newt);
            double it_score = 0;
            for (int i = 0; i < na; i++) {
                double mx = Pa[i*3]*newR[0]+Pa[i*3+1]*newR[3]+Pa[i*3+2]*newR[6]+newt[0];
                double my = Pa[i*3]*newR[1]+Pa[i*3+1]*newR[4]+Pa[i*3+2]*newR[7]+newt[1];
                double mz = Pa[i*3]*newR[2]+Pa[i*3+1]*newR[5]+Pa[i*3+2]*newR[8]+newt[2];
                double dx=mx-Qa[i*3], dy=my-Qa[i*3+1], dz=mz-Qa[i*3+2];
                double d2 = dx*dx+dy*dy+dz*dz;
                it_score += 1.0 / (1.0 + d2/d0sq_s);
            }
            it_score /= std::max(Lnorm, 1.0);
            double itR[9], itt[3];
            std::memcpy(itR, newR, 72); std::memcpy(itt, newt, 24);
            bool use_frag = !d0_search_lite;
            auto det = alignment_detailed_search(
                pred, native, ap, an, na,
                d0, d0_search, score_d8, Lnorm, N, use_frag);
            if (det.score > it_score) {
                std::memcpy(itR, det.R, 72);
                std::memcpy(itt, det.t, 24);
                it_score = det.score;
            }
            std::memcpy(curR, itR, 72); std::memcpy(curt, itt, 24);
            if (it_score > best.score) {
                best.score = it_score;
                std::memcpy(best.R, itR, 72);
                std::memcpy(best.t, itt, 24);
            }
        }
    }
    return best;
}

/* ── dp_refine wrapper ────────────────────────────────────────── */

inline SeedResult dp_refine(
    const double *pred, const double *native, const uint8_t *valid,
    const uint8_t *pv, const uint8_t *nv,
    const double *R0, const double *t0,
    double d0, double d0_search, double score_d8, double Lnorm,
    int N, int max_iter, bool d0_search_lite = false)
{
    return dp_refine_chunk(pred, native, valid, pv, nv,
                           R0, t0, d0, d0_search, score_d8, Lnorm,
                           N, max_iter, d0_search_lite);
}

#endif
