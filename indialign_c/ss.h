#ifndef INDIALIGN_SS_H
#define INDIALIGN_SS_H

#include "core.h"
#include "align.h"
#include "nw.h"
#include <cmath>
#include <cstring>
#include <vector>

/* ── RNA secondary structure from C1' geometry ───────────────── */
/* Detects stems via anti-parallel base-pair distance patterns,
   then classifies: '<' (5' stem strand), '>' (3' stem strand),
   'L' (loop/single-stranded). Uses no sequence information. */

inline void assign_rna_ss(const double *xyz, const uint8_t *v, int N,
                          char *ss) {
    for (int i = 0; i < N; i++) ss[i] = 'L';

    // Detect base pairs: C1'-C1' distance in [9.0, 12.0]A, |i-j|>3
    // Require consecutive stacking: (i,j) and (i+1,j-1) both paired
    auto d2 = [&](int a, int b) {
        double dx = xyz[a*3]-xyz[b*3], dy = xyz[a*3+1]-xyz[b*3+1],
               dz = xyz[a*3+2]-xyz[b*3+2];
        return dx*dx+dy*dy+dz*dz;
    };
    static constexpr double BP_LO2 = 9.0*9.0, BP_HI2 = 12.0*12.0;

    // Find stem regions: runs of anti-parallel base pairs
    std::vector<int> bp_partner(N, -1);
    for (int i = 0; i < N - 4; i++) {
        if (!v[i] || !v[i+1]) continue;
        for (int j = N - 1; j > i + 3; j--) {
            if (!v[j] || !v[j-1]) continue;
            double dd_ij = d2(i, j), dd_next = d2(i+1, j-1);
            if (dd_ij >= BP_LO2 && dd_ij <= BP_HI2 &&
                dd_next >= BP_LO2 && dd_next <= BP_HI2) {
                // Check no existing conflict
                if (bp_partner[i] < 0 && bp_partner[j] < 0) {
                    bp_partner[i] = j;
                    bp_partner[j] = i;
                }
            }
        }
    }

    // Assign: paired residues get '<' (lower index) or '>' (higher index)
    for (int i = 0; i < N; i++) {
        if (bp_partner[i] < 0) continue;
        ss[i] = (i < bp_partner[i]) ? '<' : '>';
    }
}

/* ── SS-only NW alignment (USalign IA2) ──────────────────────── */

inline int ss_nw_align(const char *ss_p, const char *ss_n,
                       const uint8_t *pv, const uint8_t *nv, int N,
                       int *ap, int *an) {
    std::vector<int> pv_map(N), nv_map(N);
    std::vector<char> sp(N), sn(N);
    int plen = 0, nlen = 0;
    for (int i = 0; i < N; i++)
        if (pv[i]) { pv_map[plen] = i; sp[plen++] = ss_p[i]; }
    for (int i = 0; i < N; i++)
        if (nv[i]) { nv_map[nlen] = i; sn[nlen++] = ss_n[i]; }
    if (plen < 3 || nlen < 3) return 0;

    std::vector<double> smat(plen * nlen);
    for (int pi = 0; pi < plen; pi++)
        for (int ni = 0; ni < nlen; ni++)
            smat[pi*nlen+ni] = (sp[pi] == sn[ni]) ? 1.0 : 0.0;

    std::vector<double> H;
    std::vector<int8_t> trace;
    nw_dp(smat.data(), plen, nlen, H, trace, -1.0);
    int na = nw_traceback(trace, plen, nlen, ap, an,
                          std::min(plen, nlen));
    for (int i = 0; i < na; i++) {
        ap[i] = pv_map[ap[i]];
        an[i] = nv_map[an[i]];
    }
    return na;
}

/* ── Evaluate SS seed: IA2 (pure SS) + IA4 (SS + structure) ─── */

inline SeedResult evaluate_ss_seeds(
    const double *pred, const double *native, const uint8_t *valid,
    const uint8_t *pv, const uint8_t *nv,
    const char *ss_p, const char *ss_n,
    const double *R0, const double *t0,
    double d0, double d0s, double sd8, double Lnorm, int N,
    int dp_iter)
{
    SeedResult best;
    best.score = -1e30;
    double I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(best.R,I,72);
    best.t[0]=best.t[1]=best.t[2]=0;

    std::vector<int> ap(N), an(N);

    // IA2: pure SS NW
    int na = ss_nw_align(ss_p, ss_n, pv, nv, N, ap.data(), an.data());
    if (na >= 3) {
        auto det = alignment_detailed_search(pred, native, ap.data(), an.data(), na,
                                             d0, d0s, sd8, Lnorm, N, true);
        if (det.score > best.score) best = det;
        if (dp_iter > 0) {
            auto dp = dp_refine(pred, native, valid, pv, nv,
                                best.R, best.t, d0, d0s, sd8, Lnorm,
                                N, dp_iter, false);
            if (dp.score > best.score) best = dp;
        }
    }

    // IA4: combined structure + SS scoring
    std::vector<int> pv_map(N), nv_map(N);
    int plen = 0, nlen = 0;
    for (int i = 0; i < N; i++) if (pv[i]) pv_map[plen++] = i;
    for (int i = 0; i < N; i++) if (nv[i]) nv_map[nlen++] = i;
    if (plen >= 3 && nlen >= 3) {
        const double *rR = (best.score > -1e20) ? best.R : R0;
        const double *rt = (best.score > -1e20) ? best.t : t0;
        std::vector<double> moved(N*3);
        apply_transform(pred, rR, rt, moved.data(), N);
        double d01 = std::max(d0 + 1.5, d0s);
        double d01sq = d01 * d01;
        std::vector<double> smat(plen * nlen);
        for (int pi = 0; pi < plen; pi++) {
            int ii = pv_map[pi];
            for (int ni = 0; ni < nlen; ni++) {
                int jj = nv_map[ni];
                double dx = moved[ii*3]-native[jj*3];
                double dy = moved[ii*3+1]-native[jj*3+1];
                double dz = moved[ii*3+2]-native[jj*3+2];
                double d2 = dx*dx+dy*dy+dz*dz;
                double sc = 1.0 / (1.0 + d2/d01sq);
                if (ss_p[ii] == ss_n[jj]) sc += 0.5;
                smat[pi*nlen+ni] = sc;
            }
        }
        std::vector<double> H;
        std::vector<int8_t> trace;
        nw_dp(smat.data(), plen, nlen, H, trace, -1.0);
        na = nw_traceback(trace, plen, nlen, ap.data(), an.data(),
                          std::min(plen, nlen));
        for (int i = 0; i < na; i++) {
            ap[i] = pv_map[ap[i]];
            an[i] = nv_map[an[i]];
        }
        if (na >= 3) {
            auto det = alignment_detailed_search(pred, native, ap.data(), an.data(), na,
                                                 d0, d0s, sd8, Lnorm, N, true);
            if (det.score > best.score) best = det;
            if (dp_iter > 0) {
                auto dp = dp_refine(pred, native, valid, pv, nv,
                                    best.R, best.t, d0, d0s, sd8, Lnorm,
                                    N, dp_iter, false);
                if (dp.score > best.score) best = dp;
            }
        }
    }
    return best;
}

#endif
