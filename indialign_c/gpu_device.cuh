#ifndef INDIALIGN_GPU_DEVICE_CUH
#define INDIALIGN_GPU_DEVICE_CUH

#include <cuda_runtime.h>

/* ── Warp reduction ───────────────────────────────────────────── */

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ inline float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[warp] = val;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < (unsigned)nwarps) ? warp_sums[threadIdx.x] : 0.0f;
    if (warp == 0) val = warp_reduce_sum(val);
    return val;
}

/* ── Device Jacobi 4×4 eigensolve ─────────────────────────────── */

__device__ inline void dev_jacobi4(float A[4][4], float V[4][4]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            V[i][j] = (i == j) ? 1.0f : 0.0f;
    const int ps[] = {0,0,0,1,1,2};
    const int qs[] = {1,2,3,2,3,3};
    for (int sweep = 0; sweep < 8; sweep++) {
        for (int k = 0; k < 6; k++) {
            int p = ps[k], q = qs[k];
            float apq = A[p][q];
            if (fabsf(apq) < 1e-10f) continue;
            float tau = (A[q][q] - A[p][p]) / (2.0f * apq);
            float t = copysignf(1.0f, tau) /
                      (fabsf(tau) + sqrtf(1.0f + tau * tau));
            float c = rsqrtf(1.0f + t * t), s = t * c;
            A[p][p] -= t * apq;
            A[q][q] += t * apq;
            A[p][q] = A[q][p] = 0.0f;
            for (int r = 0; r < 4; r++) {
                if (r == p || r == q) continue;
                float rp = A[r][p], rq = A[r][q];
                A[r][p] = A[p][r] = c * rp - s * rq;
                A[r][q] = A[q][r] = s * rp + c * rq;
            }
            for (int r = 0; r < 4; r++) {
                float vp = V[r][p], vq = V[r][q];
                V[r][p] = c * vp - s * vq;
                V[r][q] = s * vp + c * vq;
            }
        }
    }
}

/* ── Device Kabsch from accumulated covariance ────────────────── */

__device__ inline void dev_kabsch_from_cov(
    float H[3][3], float cP[3], float cQ[3], float count,
    float *R, float *t)
{
    if (count < 3.0f) {
        R[0]=1;R[1]=0;R[2]=0;R[3]=0;R[4]=1;R[5]=0;R[6]=0;R[7]=0;R[8]=1;
        t[0]=t[1]=t[2]=0; return;
    }
    float F[4][4], V[4][4];
    F[0][0]=H[0][0]+H[1][1]+H[2][2];
    F[0][1]=F[1][0]=H[1][2]-H[2][1];
    F[0][2]=F[2][0]=H[2][0]-H[0][2];
    F[0][3]=F[3][0]=H[0][1]-H[1][0];
    F[1][1]=H[0][0]-H[1][1]-H[2][2];
    F[1][2]=F[2][1]=H[0][1]+H[1][0];
    F[1][3]=F[3][1]=H[2][0]+H[0][2];
    F[2][2]=-H[0][0]+H[1][1]-H[2][2];
    F[2][3]=F[3][2]=H[1][2]+H[2][1];
    F[3][3]=-H[0][0]-H[1][1]+H[2][2];
    dev_jacobi4(F, V);
    int best = 0;
    for (int i = 1; i < 4; i++)
        if (F[i][i] > F[best][best]) best = i;
    float q[4];
    for (int i = 0; i < 4; i++) q[i] = V[i][best];
    float qn = sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if (qn < 1e-12f) { q[0]=1;q[1]=q[2]=q[3]=0; }
    else { float inv = 1.0f/qn; for(int i=0;i<4;i++) q[i]*=inv; }
    float q0=q[0],q1=q[1],q2=q[2],q3=q[3];
    R[0]=q0*q0+q1*q1-q2*q2-q3*q3; R[1]=2*(q1*q2+q0*q3); R[2]=2*(q1*q3-q0*q2);
    R[3]=2*(q1*q2-q0*q3); R[4]=q0*q0-q1*q1+q2*q2-q3*q3; R[5]=2*(q2*q3+q0*q1);
    R[6]=2*(q1*q3+q0*q2); R[7]=2*(q2*q3-q0*q1); R[8]=q0*q0-q1*q1-q2*q2+q3*q3;
    for (int d = 0; d < 3; d++)
        t[d] = cQ[d] - (cP[0]*R[d] + cP[1]*R[3+d] + cP[2]*R[6+d]);
}

#endif
