#ifndef INDIALIGN_SEEDS_H
#define INDIALIGN_SEEDS_H

#include <vector>
#include <set>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

static constexpr int MAX_FRAG_STARTS = 50;
static constexpr double D0_MULTS[] = {2.0, 1.5, 1.0, 0.75, 0.5, 0.25};
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
inline void build_topk_seed(const double *d2, const uint8_t *valid, int N,
                            double frac, uint8_t *mask) {
    int vc = 0;
    for (int i = 0; i < N; i++) vc += valid[i];
    int k = std::max(3, (int)(vc * frac));
    k = std::min(k, N);
    std::vector<std::pair<double,int>> dv;
    for (int i = 0; i < N; i++)
        if (valid[i]) dv.push_back({d2[i], i});
    std::sort(dv.begin(), dv.end());
    for (int i = 0; i < N; i++) mask[i] = 0;
    for (int i = 0; i < std::min(k, (int)dv.size()); i++)
        mask[dv[i].second] = 1;
}

/* ── FNV-1a hash for seed deduplication ───────────────────────── */

inline uint64_t hash_mask(const uint8_t *mask, int N) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < N; i++) {
        h ^= mask[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* ── Spatial grid for neighbor queries ────────────────────────── */

struct SpatialGrid {
    std::unordered_map<int64_t, std::vector<int>> cells;
    double cell_size;
    static constexpr int64_t P = 100003; // prime for hashing

    SpatialGrid(const double *xyz, const std::vector<int> &indices,
                double cs) : cell_size(cs) {
        for (int idx : indices) {
            int64_t key = cell_key(xyz[idx*3], xyz[idx*3+1], xyz[idx*3+2]);
            cells[key].push_back(idx);
        }
    }

    int64_t cell_key(double x, double y, double z) const {
        auto fi = [](double v, double cs) -> int64_t {
            return (int64_t)std::floor(v / cs);
        };
        int64_t ix = fi(x, cell_size), iy = fi(y, cell_size),
                iz = fi(z, cell_size);
        return (ix * P + iy) * P + iz;
    }

    // Call fn(neighbor_index) for all indices within radius of (x,y,z)
    template<typename Fn>
    void for_neighbors(double x, double y, double z, double radius,
                       Fn &&fn) const {
        double r2 = radius * radius;
        auto fi = [](double v, double cs) -> int64_t {
            return (int64_t)std::floor(v / cs);
        };
        int64_t cx = fi(x, cell_size), cy = fi(y, cell_size),
                cz = fi(z, cell_size);
        for (int64_t dx = -1; dx <= 1; dx++)
            for (int64_t dy = -1; dy <= 1; dy++)
                for (int64_t dz = -1; dz <= 1; dz++) {
                    int64_t key = ((cx+dx)*P + (cy+dy))*P + (cz+dz);
                    auto it = cells.find(key);
                    if (it == cells.end()) continue;
                    for (int j : it->second) {
                        double ddx = x - fn.xyz[j*3];
                        double ddy = y - fn.xyz[j*3+1];
                        double ddz = z - fn.xyz[j*3+2];
                        if (ddx*ddx+ddy*ddy+ddz*ddz <= r2)
                            fn(j);
                    }
                }
    }
};

/* ── Anchor contact seeds (transform-independent, spatial grid) ─ */

inline void build_anchor_contact_seeds(
    const double *pred, const double *native, const uint8_t *valid,
    int N, double d0_search, double d0, int max_anchors,
    std::vector<uint8_t> &masks, int &K)
{
    // Contact seed radii: d0s × {1.0,1.5,2.0} ∪ d0 + {1.0,2.0,3.0}
    double radii[8]; int nrad = 0;
    double mr[] = {d0_search, d0_search*1.5, d0_search*2.0};
    double ar[] = {d0+1.0, d0+2.0, d0+3.0};
    for (int i = 0; i < 3; i++) radii[nrad++] = mr[i];
    for (int i = 0; i < 3; i++) {
        bool dup = false;
        for (int j = 0; j < nrad && !dup; j++)
            dup = (std::abs(radii[j] - ar[i]) < 0.01);
        if (!dup) radii[nrad++] = ar[i];
    }

    std::vector<int> vidx;
    for (int i = 0; i < N; i++) if (valid[i]) vidx.push_back(i);
    int V = (int)vidx.size();
    if (V < 3) return;

    // Collect existing seed hashes for dedup
    std::unordered_set<uint64_t> seen_hashes;
    for (int k = 0; k < K; k++)
        seen_hashes.insert(hash_mask(&masks[k * N], N));

    for (int ri = 0; ri < nrad; ri++) {
        double rad = radii[ri];
        if (rad <= 0) continue;
        double rad_sq = rad * rad;
        double tol = std::max(0.5, rad * 0.25);

        // Build spatial grid over pred coordinates for this radius
        SpatialGrid grid(pred, vidx, rad);

        // Count consistent neighbors per valid residue using grid
        std::vector<int> counts(V, 0);
        for (int a = 0; a < V; a++) {
            int i = vidx[a];
            counts[a] = 1; // self
            // Query grid for pred-neighbors of residue i
            int64_t cx = (int64_t)std::floor(pred[i*3] / rad);
            int64_t cy = (int64_t)std::floor(pred[i*3+1] / rad);
            int64_t cz = (int64_t)std::floor(pred[i*3+2] / rad);
            for (int64_t dx = -1; dx <= 1; dx++)
                for (int64_t dy = -1; dy <= 1; dy++)
                    for (int64_t dz = -1; dz <= 1; dz++) {
                        int64_t key = ((cx+dx)*SpatialGrid::P +
                                       (cy+dy))*SpatialGrid::P + (cz+dz);
                        auto it = grid.cells.find(key);
                        if (it == grid.cells.end()) continue;
                        for (int j : it->second) {
                            if (j == i) continue;
                            double pdx=pred[i*3]-pred[j*3],
                                   pdy=pred[i*3+1]-pred[j*3+1],
                                   pdz=pred[i*3+2]-pred[j*3+2];
                            double pd2 = pdx*pdx+pdy*pdy+pdz*pdz;
                            if (pd2 > rad_sq) continue;
                            double ndx=native[i*3]-native[j*3],
                                   ndy=native[i*3+1]-native[j*3+1],
                                   ndz=native[i*3+2]-native[j*3+2];
                            double nd2 = ndx*ndx+ndy*ndy+ndz*ndz;
                            if (nd2 > rad_sq) continue;
                            if (std::abs(std::sqrt(pd2)-std::sqrt(nd2)) <= tol)
                                counts[a]++;
                        }
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
            // Rebuild neighborhood using grid
            int64_t cx2 = (int64_t)std::floor(pred[i*3] / rad);
            int64_t cy2 = (int64_t)std::floor(pred[i*3+1] / rad);
            int64_t cz2 = (int64_t)std::floor(pred[i*3+2] / rad);
            mask[i] = 1; mc = 1;
            for (int64_t dx = -1; dx <= 1; dx++)
                for (int64_t dy = -1; dy <= 1; dy++)
                    for (int64_t dz = -1; dz <= 1; dz++) {
                        int64_t key = ((cx2+dx)*SpatialGrid::P +
                                       (cy2+dy))*SpatialGrid::P + (cz2+dz);
                        auto it = grid.cells.find(key);
                        if (it == grid.cells.end()) continue;
                        for (int j : it->second) {
                            if (j == i) continue;
                            double pdx=pred[i*3]-pred[j*3],
                                   pdy=pred[i*3+1]-pred[j*3+1],
                                   pdz=pred[i*3+2]-pred[j*3+2];
                            double pd2 = pdx*pdx+pdy*pdy+pdz*pdz;
                            if (pd2 > rad_sq) continue;
                            double ndx=native[i*3]-native[j*3],
                                   ndy=native[i*3+1]-native[j*3+1],
                                   ndz=native[i*3+2]-native[j*3+2];
                            double nd2 = ndx*ndx+ndy*ndy+ndz*ndz;
                            if (nd2 > rad_sq) continue;
                            if (std::abs(std::sqrt(pd2)-std::sqrt(nd2)) <= tol)
                                { mask[j] = 1; mc++; }
                        }
                    }
            if (mc < 3) continue;
            // Hash-based deduplication
            uint64_t h = hash_mask(mask.data(), N);
            if (seen_hashes.count(h)) continue;
            seen_hashes.insert(h);
            masks.insert(masks.end(), mask.begin(), mask.end());
            K++; added++;
        }
    }
}

#endif
