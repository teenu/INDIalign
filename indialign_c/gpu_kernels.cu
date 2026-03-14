#include "gpu_device.cuh"
#include "search.h"
#include <cstdio>

static bool g_gpu_ok = false;
static bool g_gpu_checked = false;

/* ── Fused iterative refinement + TM-score kernel ─────────────── */

__global__ void refine_and_score_kernel(
    const float* __restrict__ pred,     // [N*3]
    const float* __restrict__ native,   // [N*3]
    const uint8_t* __restrict__ valid,  // [N]
    const uint8_t* __restrict__ seeds,  // [K*N]
    const float* __restrict__ d0_cands, // [C]
    int N, int K, int C, int max_iter,
    float d0, float score_d8, float Lnorm,
    float* __restrict__ R_out,          // [K*C*9]
    float* __restrict__ t_out,          // [K*C*3]
    float* __restrict__ scores)         // [K*C]
{
    int km = blockIdx.x;
    int KM = K * C;
    if (km >= KM) return;
    int kid = km / C, cid = km % C;
    int tid = threadIdx.x;
    int BS = blockDim.x;

    float d0s = d0_cands[cid];
    float d0s_sq = d0s * d0s;

    // Shared memory for Kabsch accumulators
    __shared__ float s_R[9], s_t[3];
    __shared__ float s_cP[3], s_cQ[3], s_H[9], s_cnt;

    // --- Initial Kabsch from seed mask ---
    float lcP[3]={}, lcQ[3]={}, lcnt=0;
    for (int i = tid; i < N; i += BS) {
        if (seeds[kid*N+i] && valid[i]) {
            lcP[0]+=pred[i*3]; lcP[1]+=pred[i*3+1]; lcP[2]+=pred[i*3+2];
            lcQ[0]+=native[i*3]; lcQ[1]+=native[i*3+1]; lcQ[2]+=native[i*3+2];
            lcnt += 1.0f;
        }
    }
    for (int d = 0; d < 3; d++) {
        lcP[d] = block_reduce_sum(lcP[d]);
        lcQ[d] = block_reduce_sum(lcQ[d]);
    }
    lcnt = block_reduce_sum(lcnt);
    if (tid == 0) {
        s_cnt = lcnt;
        float inv = (lcnt >= 3.0f) ? (1.0f / lcnt) : 0.0f;
        for (int d = 0; d < 3; d++) { s_cP[d] = lcP[d]*inv; s_cQ[d] = lcQ[d]*inv; }
        for (int i = 0; i < 9; i++) s_H[i] = 0;
    }
    __syncthreads();

    // Covariance
    float lH[9] = {};
    if (s_cnt >= 3.0f) {
        for (int i = tid; i < N; i += BS) {
            if (!(seeds[kid*N+i] && valid[i])) continue;
            float dp[3], dq[3];
            for (int d = 0; d < 3; d++) {
                dp[d] = pred[i*3+d] - s_cP[d];
                dq[d] = native[i*3+d] - s_cQ[d];
            }
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    lH[r*3+c] += dp[r] * dq[c];
        }
    }
    for (int i = 0; i < 9; i++) {
        float v = block_reduce_sum(lH[i]);
        if (tid == 0) s_H[i] = v;
    }
    __syncthreads();

    if (tid == 0) {
        float H33[3][3]; for (int i = 0; i < 9; i++) H33[i/3][i%3] = s_H[i];
        dev_kabsch_from_cov(H33, s_cP, s_cQ, s_cnt, s_R, s_t);
    }
    __syncthreads();

    // --- 20 iterations of refine ---
    for (int iter = 0; iter < max_iter; iter++) {
        float lcP2[3]={}, lcQ2[3]={}, lcnt2=0;
        for (int i = tid; i < N; i += BS) {
            if (!valid[i]) continue;
            float mx = pred[i*3]*s_R[0]+pred[i*3+1]*s_R[3]+pred[i*3+2]*s_R[6]+s_t[0];
            float my = pred[i*3]*s_R[1]+pred[i*3+1]*s_R[4]+pred[i*3+2]*s_R[7]+s_t[1];
            float mz = pred[i*3]*s_R[2]+pred[i*3+1]*s_R[5]+pred[i*3+2]*s_R[8]+s_t[2];
            float dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
            float d2 = dx*dx+dy*dy+dz*dz;
            if (d2 <= d0s_sq) {
                lcP2[0]+=pred[i*3]; lcP2[1]+=pred[i*3+1]; lcP2[2]+=pred[i*3+2];
                lcQ2[0]+=native[i*3]; lcQ2[1]+=native[i*3+1]; lcQ2[2]+=native[i*3+2];
                lcnt2 += 1.0f;
            }
        }
        for (int d = 0; d < 3; d++) {
            lcP2[d] = block_reduce_sum(lcP2[d]);
            lcQ2[d] = block_reduce_sum(lcQ2[d]);
        }
        lcnt2 = block_reduce_sum(lcnt2);
        if (tid == 0) s_cnt = lcnt2;
        __syncthreads();
        if (s_cnt < 3.0f) { __syncthreads(); continue; }

        if (tid == 0) {
            float inv = 1.0f / s_cnt;
            for (int d = 0; d < 3; d++) { s_cP[d] = lcP2[d]*inv; s_cQ[d] = lcQ2[d]*inv; }
            for (int i = 0; i < 9; i++) s_H[i] = 0;
        }
        __syncthreads();

        float lH2[9] = {};
        for (int i = tid; i < N; i += BS) {
            if (!valid[i]) continue;
            float mx = pred[i*3]*s_R[0]+pred[i*3+1]*s_R[3]+pred[i*3+2]*s_R[6]+s_t[0];
            float my = pred[i*3]*s_R[1]+pred[i*3+1]*s_R[4]+pred[i*3+2]*s_R[7]+s_t[1];
            float mz = pred[i*3]*s_R[2]+pred[i*3+1]*s_R[5]+pred[i*3+2]*s_R[8]+s_t[2];
            float dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
            if (dx*dx+dy*dy+dz*dz > d0s_sq) continue;
            float dp[3], dq[3];
            for (int d = 0; d < 3; d++) {
                dp[d] = pred[i*3+d] - s_cP[d];
                dq[d] = native[i*3+d] - s_cQ[d];
            }
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    lH2[r*3+c] += dp[r] * dq[c];
        }
        for (int i = 0; i < 9; i++) {
            float v = block_reduce_sum(lH2[i]);
            if (tid == 0) s_H[i] = v;
        }
        __syncthreads();
        if (tid == 0) {
            float H33[3][3]; for (int i = 0; i < 9; i++) H33[i/3][i%3] = s_H[i];
            dev_kabsch_from_cov(H33, s_cP, s_cQ, s_cnt, s_R, s_t);
        }
        __syncthreads();
    }

    // --- TM-score ---
    float d0sq = fmaxf(d0*d0, 1e-12f), sd8sq = score_d8*score_d8;
    float lscore = 0;
    for (int i = tid; i < N; i += BS) {
        if (!valid[i]) continue;
        float mx = pred[i*3]*s_R[0]+pred[i*3+1]*s_R[3]+pred[i*3+2]*s_R[6]+s_t[0];
        float my = pred[i*3]*s_R[1]+pred[i*3+1]*s_R[4]+pred[i*3+2]*s_R[7]+s_t[1];
        float mz = pred[i*3]*s_R[2]+pred[i*3+1]*s_R[5]+pred[i*3+2]*s_R[8]+s_t[2];
        float dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
        float d2 = dx*dx+dy*dy+dz*dz;
        if (d2 <= sd8sq) lscore += 1.0f / (1.0f + d2/d0sq);
    }
    lscore = block_reduce_sum(lscore);
    if (tid == 0) {
        scores[km] = lscore / fmaxf(Lnorm, 1.0f);
        for (int i = 0; i < 9; i++) R_out[km*9+i] = s_R[i];
        for (int i = 0; i < 3; i++) t_out[km*3+i] = s_t[i];
    }
}

/* ── Host dispatch ────────────────────────────────────────────── */

bool gpu_available() {
    if (!g_gpu_checked) {
        int count = 0;
        g_gpu_ok = (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
        g_gpu_checked = true;
    }
    return g_gpu_ok;
}

SeedResult gpu_evaluate_seed_bank(
    const float *pred, const float *native, const uint8_t *valid,
    const uint8_t *seeds, int K,
    const float *d0_cands, int C,
    float d0, float score_d8, float Lnorm, int N, int max_iter)
{
    int KM = K * C;
    float *d_pred, *d_native, *d_d0c, *d_R, *d_t, *d_scores;
    uint8_t *d_valid, *d_seeds;
    cudaMalloc(&d_pred,   N*3*sizeof(float));
    cudaMalloc(&d_native, N*3*sizeof(float));
    cudaMalloc(&d_valid,  N);
    cudaMalloc(&d_seeds,  K*N);
    cudaMalloc(&d_d0c,    C*sizeof(float));
    cudaMalloc(&d_R,      KM*9*sizeof(float));
    cudaMalloc(&d_t,      KM*3*sizeof(float));
    cudaMalloc(&d_scores, KM*sizeof(float));

    cudaMemcpy(d_pred,   pred,    N*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_native, native,  N*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid,  valid,   N,                 cudaMemcpyHostToDevice);
    cudaMemcpy(d_seeds,  seeds,   K*N,               cudaMemcpyHostToDevice);
    cudaMemcpy(d_d0c,    d0_cands,C*sizeof(float),   cudaMemcpyHostToDevice);

    int bs = 128;
    if (N > 256) bs = 256;
    refine_and_score_kernel<<<KM, bs>>>(
        d_pred, d_native, d_valid, d_seeds, d_d0c,
        N, K, C, max_iter, d0, score_d8, Lnorm,
        d_R, d_t, d_scores);

    std::vector<float> h_scores(KM), h_R(KM*9), h_t(KM*3);
    cudaMemcpy(h_scores.data(), d_scores, KM*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R.data(),      d_R,      KM*9*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t.data(),      d_t,      KM*3*sizeof(float), cudaMemcpyDeviceToHost);

    SeedResult best; best.score = -1e30f;
    for (int i = 0; i < KM; i++) {
        if (h_scores[i] > best.score) {
            best.score = h_scores[i];
            std::memcpy(best.R, &h_R[i*9], 36);
            std::memcpy(best.t, &h_t[i*3], 12);
        }
    }
    cudaFree(d_pred); cudaFree(d_native); cudaFree(d_valid);
    cudaFree(d_seeds); cudaFree(d_d0c);
    cudaFree(d_R); cudaFree(d_t); cudaFree(d_scores);
    return best;
}
