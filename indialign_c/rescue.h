#ifndef INDIALIGN_RESCUE_H
#define INDIALIGN_RESCUE_H

#include "core.h"
#include "align.h"
#include <vector>
#include <algorithm>
#include <cstring>

/* ── Local fragment schedule (USalign-style) ──────────────────── */

inline std::vector<int> local_frag_lengths(int plen, int nlen, bool dense) {
    int ml = std::min(plen, nlen);
    if (ml < 4) return {};
    std::vector<int> out;
    if (dense) {
        int vals[] = {ml, std::min(100,ml/2), std::min(32,ml),
                      std::min(24,ml), std::min(20,ml/3), 16, 12, 8, 4};
        for (int v : vals)
            if (v >= 4 && std::find(out.begin(), out.end(), v) == out.end())
                out.push_back(v);
    } else {
        int a = std::min(20, ml/3), b = std::min(100, ml/2);
        if (a >= 4) out.push_back(a);
        if (b >= 4 && b != a) out.push_back(b);
    }
    return out;
}

inline int local_jump(int length, bool dense) {
    int j;
    if (dense) {
        if (length > 300) j = 15;
        else if (length > 200) j = 10;
        else j = 5;
        return std::max(1, std::min(j, std::max(1, length/4)));
    }
    if (length > 250) j = 45;
    else if (length > 200) j = 35;
    else if (length > 150) j = 25;
    else j = 15;
    return std::max(1, std::min(j, std::max(1, length/3)));
}

/* ── Evaluate local fragment DP seeds (OpenMP parallel) ──────── */

inline SeedResult evaluate_local_fragment_dp(
    const double *pred, const double *native, const uint8_t *valid,
    const uint8_t *pv, const uint8_t *nv,
    double d0, double d0_search, double Lnorm, int N, bool dense)
{
    SeedResult best;
    best.score = -1e30;
    double I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(best.R,I,72);
    best.t[0]=best.t[1]=best.t[2]=0;

    int plen = 0, nlen = 0;
    for (int i = 0; i < N; i++) { plen += pv[i]; nlen += nv[i]; }
    auto flens = local_frag_lengths(plen, nlen, dense);
    if (flens.empty()) return best;
    int jp = local_jump(plen, dense), jn = local_jump(nlen, dense);

    // Flatten work items for parallel dispatch
    struct FragWork { int flen, ps, ns; };
    std::vector<FragWork> work;
    for (int flen : flens) {
        int plim = plen - flen + 1, nlim = nlen - flen + 1;
        if (plim <= 0 || nlim <= 0) continue;
        for (int ps = 0; ps < plim; ps += jp)
            for (int ns = 0; ns < nlim; ns += jn)
                work.push_back({flen, ps, ns});
    }
    int nwork = (int)work.size();
    if (nwork == 0) return best;

    #pragma omp parallel
    {
        SeedResult local_best;
        local_best.score = -1e30;

        #pragma omp for schedule(dynamic)
        for (int w = 0; w < nwork; w++) {
            int flen = work[w].flen, ps = work[w].ps, ns = work[w].ns;
            std::vector<double> Pf(flen*3), Qf(flen*3);
            std::vector<uint8_t> fm(flen, 1);
            for (int i = 0; i < flen; i++) {
                std::memcpy(&Pf[i*3], &pred[(ps+i)*3], 24);
                std::memcpy(&Qf[i*3], &native[(ns+i)*3], 24);
            }
            double sR[9], st[3];
            kabsch(Pf.data(), Qf.data(), fm.data(), flen, sR, st);
            auto dp = dp_refine(pred, native, valid, pv, nv,
                                sR, st, d0, d0_search, 100.0,
                                Lnorm, N, 2, true);
            if (dp.score > local_best.score) local_best = dp;
        }

        #pragma omp critical
        { if (local_best.score > best.score) best = local_best; }
    }
    return best;
}

/* ── Evaluate threading DP seeds (OpenMP parallel) ───────────── */

inline SeedResult evaluate_threading_dp(
    const double *pred, const double *native, const uint8_t *valid,
    const uint8_t *pv, const uint8_t *nv,
    double d0, double d0_search, double Lnorm, int N)
{
    SeedResult best;
    best.score = -1e30;
    double I[9]={1,0,0,0,1,0,0,0,1}; std::memcpy(best.R,I,72);
    best.t[0]=best.t[1]=best.t[2]=0;

    int plen = 0, nlen = 0;
    for (int i = 0; i < N; i++) { plen += pv[i]; nlen += nv[i]; }
    if (std::min(plen, nlen) < 3) return best;
    int step = std::max(1, std::min(plen, nlen) / 50);

    // Collect offsets for parallel dispatch
    std::vector<int> offsets;
    for (int offset = -(nlen-3); offset < plen-2; offset += step)
        offsets.push_back(offset);
    int noff = (int)offsets.size();
    if (noff == 0) return best;

    #pragma omp parallel
    {
        SeedResult local_best;
        local_best.score = -1e30;

        #pragma omp for schedule(dynamic)
        for (int oi = 0; oi < noff; oi++) {
            int offset = offsets[oi];
            int ps = std::max(offset, 0);
            int ns = std::max(-offset, 0);
            int overlap = std::min(plen - ps, nlen - ns);
            if (overlap < 3) continue;

            int ap[4096], an[4096];
            for (int i = 0; i < overlap; i++) { ap[i] = ps+i; an[i] = ns+i; }

            auto det = alignment_detailed_search(
                pred, native, ap, an, overlap,
                d0, d0_search, 100.0, Lnorm, N, true, 8, 5);

            auto dp = dp_refine(pred, native, valid, pv, nv,
                                det.R, det.t, d0, d0_search, 100.0,
                                Lnorm, N, 1, true);
            double cand_score = std::max(det.score, dp.score);
            SeedResult cand;
            if (dp.score > det.score) {
                cand = dp;
            } else {
                cand = det;
            }
            cand.score = cand_score;
            if (cand.score > local_best.score) local_best = cand;
        }

        #pragma omp critical
        { if (local_best.score > best.score) best = local_best; }
    }
    return best;
}

#endif
