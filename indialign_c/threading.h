#ifndef INDIALIGN_THREADING_H
#define INDIALIGN_THREADING_H

#include "core.h"
#include "search.h"
#include <cstring>
#include <cmath>
#include <algorithm>

/* Same-index iterative refinement from a given R,t starting point.
   Unlike iterative_seed_refine, preserves the input R,t as starting
   point instead of overwriting with a seed-mask Kabsch. */
inline void refine_from_rt(
    const double *pred, const double *native, const uint8_t *valid,
    double d0_search, int N, int n_iter, double *R, double *t)
{
    double d0s_sq = d0_search * d0_search;
    for (int iter = 0; iter < n_iter; iter++) {
        uint8_t sel[4096]; int cnt = 0;
        for (int i = 0; i < N; i++) {
            if (!valid[i]) { sel[i] = 0; continue; }
            double mx = pred[i*3]*R[0]+pred[i*3+1]*R[3]+pred[i*3+2]*R[6]+t[0];
            double my = pred[i*3]*R[1]+pred[i*3+1]*R[4]+pred[i*3+2]*R[7]+t[1];
            double mz = pred[i*3]*R[2]+pred[i*3+1]*R[5]+pred[i*3+2]*R[8]+t[2];
            double dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
            sel[i] = (dx*dx+dy*dy+dz*dz <= d0s_sq) ? 1 : 0;
            cnt += sel[i];
        }
        if (cnt >= 3) kabsch(pred, native, sel, N, R, t);
    }
}

/* ── Gapless threading: try offset alignments pred[i] → native[i-k] ── */

inline SeedResult gapless_threading_search(
    const double *pred, const double *native,
    const uint8_t *pv, const uint8_t *nv, const uint8_t *valid,
    double d0, double d0_search, double Lnorm, int N)
{
    SeedResult best; best.score = -1e30;
    int plen = 0, nlen = 0;
    for (int i = 0; i < N; i++) { plen += pv[i]; nlen += nv[i]; }
    int min_ali = std::max(3, std::min(plen, nlen) / 2);
    int k_range = N - min_ali;
    if (k_range <= 0) return best;
    int stride = std::max(1, (2 * k_range + 99) / 100);

    double Pa[4096*3], Qa[4096*3];
    uint8_t pmask[4096];

    for (int k = -k_range; k <= k_range; k += stride) {
        if (k == 0) continue;
        int na = 0;
        for (int i = 0; i < N; i++) {
            int j = i - k;
            if (j < 0 || j >= N || !pv[i] || !nv[j]) continue;
            std::memcpy(&Pa[na*3], &pred[i*3], 24);
            std::memcpy(&Qa[na*3], &native[j*3], 24);
            pmask[na++] = 1;
            if (na >= 4096) break;
        }
        if (na < min_ali) continue;
        double R[9], t[3];
        kabsch(Pa, Qa, pmask, na, R, t);
        refine_from_rt(pred, native, valid, d0_search, N, 5, R, t);
        double sc = tm_score_no_d8(pred, native, valid, R, t, d0, Lnorm, N);
        if (sc > best.score) {
            best.score = sc;
            std::memcpy(best.R, R, 72);
            std::memcpy(best.t, t, 24);
        }
    }
    return best;
}

/* ── Cross-position fragment seeds (USalign IA5-style) ────────────── */

inline SeedResult cross_fragment_search(
    const double *pred, const double *native,
    const uint8_t *pv, const uint8_t *nv, const uint8_t *valid,
    double d0, double d0_search, double Lnorm, int N)
{
    SeedResult best; best.score = -1e30;
    int vc = 0;
    for (int i = 0; i < N; i++) vc += valid[i];
    if (vc < 6) return best;

    int frags[2] = {std::min(20, std::max(4, vc / 3)),
                    std::min(vc / 2, 100)};
    int n_frag = (frags[1] > frags[0] + 4) ? 2 : 1;
    int n_jump = (vc > 250) ? 45 : (vc > 200) ? 35 : (vc > 150) ? 25 : 15;
    n_jump = std::max(1, std::min(n_jump, vc / 3));

    double Pa[4096*3], Qa[4096*3];
    uint8_t pmask[4096];

    for (int fi = 0; fi < n_frag; fi++) {
        int flen = frags[fi];
        if (flen < 4) continue;
        for (int pi = 0; pi + flen <= N; pi += n_jump) {
            for (int ni = 0; ni + flen <= N; ni += n_jump) {
                if (pi == ni) continue;
                int na = 0;
                for (int f = 0; f < flen; f++) {
                    if (!pv[pi + f] || !nv[ni + f]) continue;
                    std::memcpy(&Pa[na*3], &pred[(pi+f)*3], 24);
                    std::memcpy(&Qa[na*3], &native[(ni+f)*3], 24);
                    pmask[na++] = 1;
                }
                if (na < 3) continue;
                double R[9], t[3];
                kabsch(Pa, Qa, pmask, na, R, t);
                refine_from_rt(pred, native, valid, d0_search, N, 5, R, t);
                double sc = tm_score_no_d8(pred, native, valid, R, t, d0, Lnorm, N);
                if (sc > best.score) {
                    best.score = sc;
                    std::memcpy(best.R, R, 72);
                    std::memcpy(best.t, t, 24);
                }
            }
        }
    }
    return best;
}

#endif
