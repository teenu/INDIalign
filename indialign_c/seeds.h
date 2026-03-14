#ifndef INDIALIGN_SEEDS_H
#define INDIALIGN_SEEDS_H

#include <vector>
#include <set>
#include <algorithm>
#include <cstdint>

static constexpr int MAX_FRAG_STARTS = 50;
static constexpr float D0_MULTS[] = {2.0f, 1.5f, 1.0f, 0.75f, 0.5f, 0.25f};
static constexpr int N_D0_MULTS = 6;

inline std::vector<int> fragment_lengths(int max_len) {
    if (max_len < 4) return {};
    std::set<int, std::greater<int>> s;
    int vals[] = {max_len, max_len/2, max_len/4, max_len/8, 64, 32, 16, 8, 4};
    for (int v : vals)
        if (v >= 4 && v <= max_len) s.insert(v);
    return {s.begin(), s.end()};
}

inline std::vector<int> start_positions(int n, int frag, int ms = MAX_FRAG_STARTS) {
    int total = n - frag + 1;
    if (total <= 0) return {};
    if (total <= ms) {
        std::vector<int> out(total);
        for (int i = 0; i < total; i++) out[i] = i;
        return out;
    }
    int step = std::max(1, total / ms);
    std::vector<int> out;
    for (int i = 0; i < total; i += step) out.push_back(i);
    if (out.back() != total - 1) out.push_back(total - 1);
    return out;
}

/* Build fragment seed masks for one pair (B=1).
   Returns K masks, each of length N.  Seeds with < 3 set bits are discarded. */
inline void build_seed_masks(int N, const uint8_t *valid, bool use_frag,
                             std::vector<uint8_t> &masks, int &K,
                             int max_frag_starts = MAX_FRAG_STARTS) {
    int vc = 0;
    for (int i = 0; i < N; i++) vc += valid[i];
    masks.clear();
    K = 0;
    // Global seed
    masks.insert(masks.end(), valid, valid + N);
    K++;
    if (use_frag) {
        auto frags = fragment_lengths(vc);
        for (int flen : frags) {
            auto starts = start_positions(N, flen, max_frag_starts);
            for (int s : starts) {
                int cnt = 0;
                for (int i = s; i < s + flen && i < N; i++) cnt += valid[i];
                if (cnt < 3) continue;
                size_t off = masks.size();
                masks.resize(off + N, 0);
                for (int i = s; i < s + flen && i < N; i++)
                    masks[off + i] = valid[i];
                K++;
            }
        }
    }
    // Filter seeds with < 3 valid residues
    std::vector<uint8_t> filtered;
    int newK = 0;
    for (int k = 0; k < K; k++) {
        int cnt = 0;
        for (int i = 0; i < N; i++) cnt += masks[k * N + i];
        if (cnt >= 3) {
            filtered.insert(filtered.end(),
                            masks.begin() + k * N,
                            masks.begin() + (k+1) * N);
            newK++;
        }
    }
    masks = std::move(filtered);
    K = newK;
}

/* Build top-k distance seed (single frac=0.5 for strict mode). */
inline void build_topk_seed(const float *d2, const uint8_t *valid, int N,
                            float frac, uint8_t *mask) {
    int vc = 0;
    for (int i = 0; i < N; i++) vc += valid[i];
    int k = std::max(3, (int)(vc * frac));
    k = std::min(k, N);
    std::vector<std::pair<float,int>> dv;
    for (int i = 0; i < N; i++)
        if (valid[i]) dv.push_back({d2[i], i});
    std::sort(dv.begin(), dv.end());
    std::memset(mask, 0, N);
    for (int i = 0; i < std::min(k, (int)dv.size()); i++)
        mask[dv[i].second] = 1;
}

/* ── Anchor contact seeds (transform-independent) ────────────── */

inline void build_anchor_contact_seeds(
    const float *pred, const float *native, const uint8_t *valid,
    int N, float d0_search, float d0, int max_anchors,
    std::vector<uint8_t> &masks, int &K)
{
    // Contact seed radii: d0s × {1.0,1.5,2.0} ∪ d0 + {1.0,2.0,3.0}
    float radii[8]; int nrad = 0;
    float mr[] = {d0_search, d0_search*1.5f, d0_search*2.0f};
    float ar[] = {d0+1.0f, d0+2.0f, d0+3.0f};
    for (int i = 0; i < 3; i++) radii[nrad++] = mr[i];
    for (int i = 0; i < 3; i++) {
        bool dup = false;
        for (int j = 0; j < nrad && !dup; j++)
            dup = (std::abs(radii[j] - ar[i]) < 0.01f);
        if (!dup) radii[nrad++] = ar[i];
    }

    std::vector<int> vidx;
    for (int i = 0; i < N; i++) if (valid[i]) vidx.push_back(i);
    int V = (int)vidx.size();
    if (V < 3) return;

    for (int ri = 0; ri < nrad; ri++) {
        float rad = radii[ri];
        if (rad <= 0) continue;
        float rad_sq = rad * rad;
        float tol = std::max(0.5f, rad * 0.25f);

        // Count consistent neighbors per valid residue
        std::vector<int> counts(V, 0);
        for (int a = 0; a < V; a++) {
            int i = vidx[a];
            counts[a] = 1; // self
            for (int b = 0; b < V; b++) {
                if (a == b) continue;
                int j = vidx[b];
                float pdx=pred[i*3]-pred[j*3], pdy=pred[i*3+1]-pred[j*3+1],
                      pdz=pred[i*3+2]-pred[j*3+2];
                float pd2 = pdx*pdx+pdy*pdy+pdz*pdz;
                if (pd2 > rad_sq) continue;
                float ndx=native[i*3]-native[j*3], ndy=native[i*3+1]-native[j*3+1],
                      ndz=native[i*3+2]-native[j*3+2];
                float nd2 = ndx*ndx+ndy*ndy+ndz*ndz;
                if (nd2 > rad_sq) continue;
                if (std::abs(std::sqrt(pd2)-std::sqrt(nd2)) <= tol)
                    counts[a]++;
            }
        }
        // Sort candidates by count descending
        std::vector<std::pair<int,int>> cands;
        for (int a = 0; a < V; a++)
            if (counts[a] >= 3) cands.push_back({counts[a], a});
        std::sort(cands.begin(), cands.end(),
                  [](auto &x, auto &y){ return x.first > y.first; });

        int added = 0;
        for (auto &[cnt, a] : cands) {
            if (added >= max_anchors) break;
            int i = vidx[a];
            std::vector<uint8_t> mask(N, 0);
            int mc = 0;
            for (int b = 0; b < V; b++) {
                int j = vidx[b];
                if (a == b) { mask[j] = 1; mc++; continue; }
                float pdx=pred[i*3]-pred[j*3], pdy=pred[i*3+1]-pred[j*3+1],
                      pdz=pred[i*3+2]-pred[j*3+2];
                float pd2 = pdx*pdx+pdy*pdy+pdz*pdz;
                if (pd2 > rad_sq) continue;
                float ndx=native[i*3]-native[j*3], ndy=native[i*3+1]-native[j*3+1],
                      ndz=native[i*3+2]-native[j*3+2];
                float nd2 = ndx*ndx+ndy*ndy+ndz*ndz;
                if (nd2 > rad_sq) continue;
                if (std::abs(std::sqrt(pd2)-std::sqrt(nd2)) <= tol)
                    { mask[j] = 1; mc++; }
            }
            if (mc < 3) continue;
            // Deduplicate against existing seeds
            bool dup = false;
            for (int k = 0; k < K && !dup; k++) {
                bool same = true;
                for (int n = 0; n < N && same; n++)
                    same = (mask[n] == masks[k*N+n]);
                dup = same;
            }
            if (dup) continue;
            masks.insert(masks.end(), mask.begin(), mask.end());
            K++; added++;
        }
    }
}

#endif
