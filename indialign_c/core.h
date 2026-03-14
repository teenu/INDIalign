#ifndef INDIALIGN_CORE_H
#define INDIALIGN_CORE_H

#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>

/* ── d0 normalization (RNA) ────────────────────────────────────── */

inline void d0_from_length_rna(float L, float *d0, float *d0s, float *sd8) {
    float Lc = std::max(L, 1.0f);
    if      (Lc <= 11.0f) *d0 = 0.3f;
    else if (Lc <= 15.0f) *d0 = 0.4f;
    else if (Lc <= 19.0f) *d0 = 0.5f;
    else if (Lc <= 23.0f) *d0 = 0.6f;
    else                  *d0 = 0.7f;
    if (Lc >= 30.0f)
        *d0 = 0.6f * std::sqrt(Lc - 0.5f) - 2.5f;
    if (*d0 < 0.3f) *d0 = 0.3f;
    float v = 1.24f * std::cbrt(std::max(Lc - 15.0f, 0.0f)) - 1.8f;
    *d0s = std::clamp(v, 4.5f, 8.0f);
    *sd8 = 1.5f * std::pow(Lc, 0.3f) + 3.5f;
}

/* ── Jacobi 4×4 symmetric eigensolve ──────────────────────────── */

inline void jacobi4(double A[4][4], double V[4][4]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            V[i][j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 8; sweep++) {
        static const int ps[] = {0,0,0,1,1,2};
        static const int qs[] = {1,2,3,2,3,3};
        for (int k = 0; k < 6; k++) {
            int p = ps[k], q = qs[k];
            double apq = A[p][q];
            if (std::abs(apq) < 1e-15) continue;
            double tau = (A[q][q] - A[p][p]) / (2.0 * apq);
            double t = (tau >= 0 ? 1.0 : -1.0) /
                       (std::abs(tau) + std::sqrt(1.0 + tau * tau));
            double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
            A[p][p] -= t * apq;
            A[q][q] += t * apq;
            A[p][q] = A[q][p] = 0.0;
            for (int r = 0; r < 4; r++) {
                if (r == p || r == q) continue;
                double rp = A[r][p], rq = A[r][q];
                A[r][p] = A[p][r] = c * rp - s * rq;
                A[r][q] = A[q][r] = s * rp + c * rq;
            }
            for (int r = 0; r < 4; r++) {
                double vp = V[r][p], vq = V[r][q];
                V[r][p] = c * vp - s * vq;
                V[r][q] = s * vp + c * vq;
            }
        }
    }
}

/* ── Kabsch alignment (Horn quaternion) ───────────────────────── */

inline void kabsch(const float *P, const float *Q,
                   const uint8_t *mask, int N, float *R, float *t) {
    double wsum = 0, cP[3] = {}, cQ[3] = {};
    for (int i = 0; i < N; i++) {
        if (!mask[i]) continue;
        for (int d = 0; d < 3; d++) {
            cP[d] += P[i*3+d]; cQ[d] += Q[i*3+d];
        }
        wsum += 1.0;
    }
    if (wsum < 3.0) {
        float I[9] = {1,0,0, 0,1,0, 0,0,1};
        std::memcpy(R, I, 36); t[0]=t[1]=t[2]=0; return;
    }
    for (int d = 0; d < 3; d++) { cP[d] /= wsum; cQ[d] /= wsum; }
    double H[3][3] = {};
    for (int i = 0; i < N; i++) {
        if (!mask[i]) continue;
        double dp[3], dq[3];
        for (int d = 0; d < 3; d++) {
            dp[d] = P[i*3+d] - cP[d]; dq[d] = Q[i*3+d] - cQ[d];
        }
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                H[r][c] += dp[r] * dq[c];
    }
    double F[4][4] = {};
    F[0][0] = H[0][0]+H[1][1]+H[2][2];
    F[0][1]=F[1][0] = H[1][2]-H[2][1];
    F[0][2]=F[2][0] = H[2][0]-H[0][2];
    F[0][3]=F[3][0] = H[0][1]-H[1][0];
    F[1][1] = H[0][0]-H[1][1]-H[2][2];
    F[1][2]=F[2][1] = H[0][1]+H[1][0];
    F[1][3]=F[3][1] = H[2][0]+H[0][2];
    F[2][2] = -H[0][0]+H[1][1]-H[2][2];
    F[2][3]=F[3][2] = H[1][2]+H[2][1];
    F[3][3] = -H[0][0]-H[1][1]+H[2][2];
    double V[4][4];
    jacobi4(F, V);
    int best = 0;
    for (int i = 1; i < 4; i++)
        if (F[i][i] > F[best][best]) best = i;
    double q[4];
    for (int i = 0; i < 4; i++) q[i] = V[i][best];
    double qn = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if (qn < 1e-12) { q[0]=1; q[1]=q[2]=q[3]=0; }
    else { for (int i = 0; i < 4; i++) q[i] /= qn; }
    float q0=q[0],q1=q[1],q2=q[2],q3=q[3];
    R[0]=q0*q0+q1*q1-q2*q2-q3*q3; R[1]=2*(q1*q2+q0*q3); R[2]=2*(q1*q3-q0*q2);
    R[3]=2*(q1*q2-q0*q3); R[4]=q0*q0-q1*q1+q2*q2-q3*q3; R[5]=2*(q2*q3+q0*q1);
    R[6]=2*(q1*q3+q0*q2); R[7]=2*(q2*q3-q0*q1); R[8]=q0*q0-q1*q1-q2*q2+q3*q3;
    for (int d = 0; d < 3; d++)
        t[d] = (float)(cQ[d] - (cP[0]*R[0*3+d]+cP[1]*R[1*3+d]+cP[2]*R[2*3+d]));
}

/* ── Transform & TM-score ─────────────────────────────────────── */

inline void apply_transform(const float *P, const float *R, const float *t,
                            float *out, int N) {
    for (int i = 0; i < N; i++)
        for (int d = 0; d < 3; d++)
            out[i*3+d] = P[i*3]*R[d] + P[i*3+1]*R[3+d] + P[i*3+2]*R[6+d] + t[d];
}

inline float tm_score(const float *pred, const float *native,
                      const uint8_t *valid, const float *R, const float *t,
                      float d0, float Lnorm, float score_d8, int N) {
    float d0sq = std::max(d0*d0, 1e-12f), sd8sq = score_d8*score_d8;
    float sum = 0;
    for (int i = 0; i < N; i++) {
        if (!valid[i]) continue;
        float mx = pred[i*3]*R[0]+pred[i*3+1]*R[3]+pred[i*3+2]*R[6]+t[0];
        float my = pred[i*3]*R[1]+pred[i*3+1]*R[4]+pred[i*3+2]*R[7]+t[1];
        float mz = pred[i*3]*R[2]+pred[i*3+1]*R[5]+pred[i*3+2]*R[8]+t[2];
        float dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
        float d2 = dx*dx+dy*dy+dz*dz;
        if (d2 <= sd8sq)
            sum += 1.0f / (1.0f + d2/d0sq);
    }
    return sum / std::max(Lnorm, 1.0f);
}

inline float tm_score_no_d8(const float *pred, const float *native,
                            const uint8_t *valid, const float *R, const float *t,
                            float d0, float Lnorm, int N) {
    float d0sq = std::max(d0*d0, 1e-12f), sum = 0;
    for (int i = 0; i < N; i++) {
        if (!valid[i]) continue;
        float mx = pred[i*3]*R[0]+pred[i*3+1]*R[3]+pred[i*3+2]*R[6]+t[0];
        float my = pred[i*3]*R[1]+pred[i*3+1]*R[4]+pred[i*3+2]*R[7]+t[1];
        float mz = pred[i*3]*R[2]+pred[i*3+1]*R[5]+pred[i*3+2]*R[8]+t[2];
        float dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
        float d2 = dx*dx+dy*dy+dz*dz;
        sum += 1.0f / (1.0f + d2/d0sq);
    }
    return sum / std::max(Lnorm, 1.0f);
}

inline void dist2_fused(const float *pred, const float *native,
                        const uint8_t *valid, const float *R, const float *t,
                        float *d2out, int N) {
    for (int i = 0; i < N; i++) {
        if (!valid[i]) { d2out[i] = 1e12f; continue; }
        float mx = pred[i*3]*R[0]+pred[i*3+1]*R[3]+pred[i*3+2]*R[6]+t[0];
        float my = pred[i*3]*R[1]+pred[i*3+1]*R[4]+pred[i*3+2]*R[7]+t[1];
        float mz = pred[i*3]*R[2]+pred[i*3+1]*R[5]+pred[i*3+2]*R[8]+t[2];
        float dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
        d2out[i] = dx*dx+dy*dy+dz*dz;
    }
}

/* ── Weighted Kabsch (continuous weights) ─────────────────────── */

inline void kabsch_weighted(const float *P, const float *Q,
                            const uint8_t *valid, const float *w,
                            int N, float *R, float *t) {
    double wsum = 0, cP[3] = {}, cQ[3] = {};
    for (int i = 0; i < N; i++) {
        if (!valid[i] || w[i] <= 0) continue;
        double wi = w[i];
        for (int d = 0; d < 3; d++) {
            cP[d] += wi * P[i*3+d]; cQ[d] += wi * Q[i*3+d];
        }
        wsum += wi;
    }
    if (wsum < 1e-12) {
        float I[9] = {1,0,0, 0,1,0, 0,0,1};
        std::memcpy(R, I, 36); t[0]=t[1]=t[2]=0; return;
    }
    for (int d = 0; d < 3; d++) { cP[d] /= wsum; cQ[d] /= wsum; }
    double H[3][3] = {};
    for (int i = 0; i < N; i++) {
        if (!valid[i] || w[i] <= 0) continue;
        double wi = w[i], dp[3], dq[3];
        for (int d = 0; d < 3; d++) {
            dp[d] = P[i*3+d] - cP[d]; dq[d] = Q[i*3+d] - cQ[d];
        }
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                H[r][c] += wi * dp[r] * dq[c];
    }
    double F[4][4] = {};
    F[0][0] = H[0][0]+H[1][1]+H[2][2];
    F[0][1]=F[1][0] = H[1][2]-H[2][1];
    F[0][2]=F[2][0] = H[2][0]-H[0][2];
    F[0][3]=F[3][0] = H[0][1]-H[1][0];
    F[1][1] = H[0][0]-H[1][1]-H[2][2];
    F[1][2]=F[2][1] = H[0][1]+H[1][0];
    F[1][3]=F[3][1] = H[2][0]+H[0][2];
    F[2][2] = -H[0][0]+H[1][1]-H[2][2];
    F[2][3]=F[3][2] = H[1][2]+H[2][1];
    F[3][3] = -H[0][0]-H[1][1]+H[2][2];
    double V[4][4];
    jacobi4(F, V);
    int best = 0;
    for (int i = 1; i < 4; i++)
        if (F[i][i] > F[best][best]) best = i;
    double q[4];
    for (int i = 0; i < 4; i++) q[i] = V[i][best];
    double qn = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if (qn < 1e-12) { q[0]=1; q[1]=q[2]=q[3]=0; }
    else { for (int i = 0; i < 4; i++) q[i] /= qn; }
    float q0=q[0],q1=q[1],q2=q[2],q3=q[3];
    R[0]=q0*q0+q1*q1-q2*q2-q3*q3; R[1]=2*(q1*q2+q0*q3); R[2]=2*(q1*q3-q0*q2);
    R[3]=2*(q1*q2-q0*q3); R[4]=q0*q0-q1*q1+q2*q2-q3*q3; R[5]=2*(q2*q3+q0*q1);
    R[6]=2*(q1*q3+q0*q2); R[7]=2*(q2*q3-q0*q1); R[8]=q0*q0-q1*q1-q2*q2+q3*q3;
    for (int d = 0; d < 3; d++)
        t[d] = (float)(cQ[d] - (cP[0]*R[0*3+d]+cP[1]*R[1*3+d]+cP[2]*R[2*3+d]));
}

/* ── Weighted TM refinement ──────────────────────────────────── */

inline void weighted_tm_refine(const float *pred, const float *native,
                               const uint8_t *valid, float d0, float Lnorm,
                               int N, int n_iters, float *R, float *t,
                               float *out_score) {
    float d0sq = std::max(d0*d0, 1e-12f);
    float weights[4096];
    for (int iter = 0; iter < n_iters; iter++) {
        for (int i = 0; i < N; i++) {
            if (!valid[i]) { weights[i] = 0; continue; }
            float mx = pred[i*3]*R[0]+pred[i*3+1]*R[3]+pred[i*3+2]*R[6]+t[0];
            float my = pred[i*3]*R[1]+pred[i*3+1]*R[4]+pred[i*3+2]*R[7]+t[1];
            float mz = pred[i*3]*R[2]+pred[i*3+1]*R[5]+pred[i*3+2]*R[8]+t[2];
            float dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
            float d2 = dx*dx+dy*dy+dz*dz;
            weights[i] = 1.0f / (1.0f + d2/d0sq);
        }
        kabsch_weighted(pred, native, valid, weights, N, R, t);
    }
    *out_score = tm_score_no_d8(pred, native, valid, R, t, d0, Lnorm, N);
}

#endif
