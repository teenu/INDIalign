#ifndef INDIALIGN_CORE_H
#define INDIALIGN_CORE_H

#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>

/* ── d0 normalization (RNA) ────────────────────────────────────── */

inline void d0_from_length_rna(double L, double *d0, double *d0s, double *sd8) {
    double Lc = std::max(L, 1.0);
    if      (Lc <= 11.0) *d0 = 0.3;
    else if (Lc <= 15.0) *d0 = 0.4;
    else if (Lc <= 19.0) *d0 = 0.5;
    else if (Lc <= 23.0) *d0 = 0.6;
    else                  *d0 = 0.7;
    if (Lc >= 30.0)
        *d0 = 0.6 * std::sqrt(Lc - 0.5) - 2.5;
    if (*d0 < 0.3) *d0 = 0.3;
    double v = 1.24 * std::cbrt(std::max(Lc - 15.0, 0.0)) - 1.8;
    *d0s = std::clamp(v, 4.5, 8.0);
    *sd8 = 1.5 * std::pow(Lc, 0.3) + 3.5;
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

inline void kabsch(const double *P, const double *Q,
                   const uint8_t *mask, int N, double *R, double *t) {
    double wsum = 0, cP[3] = {}, cQ[3] = {};
    for (int i = 0; i < N; i++) {
        if (!mask[i]) continue;
        for (int d = 0; d < 3; d++) {
            cP[d] += P[i*3+d]; cQ[d] += Q[i*3+d];
        }
        wsum += 1.0;
    }
    if (wsum < 3.0) {
        double I[9] = {1,0,0, 0,1,0, 0,0,1};
        std::memcpy(R, I, 72); t[0]=t[1]=t[2]=0; return;
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
    double q0=q[0],q1=q[1],q2=q[2],q3=q[3];
    R[0]=q0*q0+q1*q1-q2*q2-q3*q3; R[1]=2*(q1*q2+q0*q3); R[2]=2*(q1*q3-q0*q2);
    R[3]=2*(q1*q2-q0*q3); R[4]=q0*q0-q1*q1+q2*q2-q3*q3; R[5]=2*(q2*q3+q0*q1);
    R[6]=2*(q1*q3+q0*q2); R[7]=2*(q2*q3-q0*q1); R[8]=q0*q0-q1*q1-q2*q2+q3*q3;
    for (int d = 0; d < 3; d++)
        t[d] = cQ[d] - (cP[0]*R[0*3+d]+cP[1]*R[1*3+d]+cP[2]*R[2*3+d]);
}

/* ── Transform & TM-score ─────────────────────────────────────── */

inline void apply_transform(const double *P, const double *R, const double *t,
                            double *out, int N) {
    for (int i = 0; i < N; i++)
        for (int d = 0; d < 3; d++)
            out[i*3+d] = P[i*3]*R[d] + P[i*3+1]*R[3+d] + P[i*3+2]*R[6+d] + t[d];
}

inline double tm_score(const double *pred, const double *native,
                       const uint8_t *valid, const double *R, const double *t,
                       double d0, double Lnorm, double score_d8, int N) {
    double d0sq = std::max(d0*d0, 1e-12), sd8sq = score_d8*score_d8;
    double sum = 0;
    for (int i = 0; i < N; i++) {
        if (!valid[i]) continue;
        double mx = pred[i*3]*R[0]+pred[i*3+1]*R[3]+pred[i*3+2]*R[6]+t[0];
        double my = pred[i*3]*R[1]+pred[i*3+1]*R[4]+pred[i*3+2]*R[7]+t[1];
        double mz = pred[i*3]*R[2]+pred[i*3+1]*R[5]+pred[i*3+2]*R[8]+t[2];
        double dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
        double d2 = dx*dx+dy*dy+dz*dz;
        if (d2 <= sd8sq)
            sum += 1.0 / (1.0 + d2/d0sq);
    }
    return sum / std::max(Lnorm, 1.0);
}

inline double tm_score_no_d8(const double *pred, const double *native,
                             const uint8_t *valid, const double *R, const double *t,
                             double d0, double Lnorm, int N) {
    double d0sq = std::max(d0*d0, 1e-12), sum = 0;
    for (int i = 0; i < N; i++) {
        if (!valid[i]) continue;
        double mx = pred[i*3]*R[0]+pred[i*3+1]*R[3]+pred[i*3+2]*R[6]+t[0];
        double my = pred[i*3]*R[1]+pred[i*3+1]*R[4]+pred[i*3+2]*R[7]+t[1];
        double mz = pred[i*3]*R[2]+pred[i*3+1]*R[5]+pred[i*3+2]*R[8]+t[2];
        double dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
        double d2 = dx*dx+dy*dy+dz*dz;
        sum += 1.0 / (1.0 + d2/d0sq);
    }
    return sum / std::max(Lnorm, 1.0);
}

inline void dist2_fused(const double *pred, const double *native,
                        const uint8_t *valid, const double *R, const double *t,
                        double *d2out, int N) {
    for (int i = 0; i < N; i++) {
        if (!valid[i]) { d2out[i] = 1e12; continue; }
        double mx = pred[i*3]*R[0]+pred[i*3+1]*R[3]+pred[i*3+2]*R[6]+t[0];
        double my = pred[i*3]*R[1]+pred[i*3+1]*R[4]+pred[i*3+2]*R[7]+t[1];
        double mz = pred[i*3]*R[2]+pred[i*3+1]*R[5]+pred[i*3+2]*R[8]+t[2];
        double dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
        d2out[i] = dx*dx+dy*dy+dz*dz;
    }
}

/* ── Weighted Kabsch (continuous weights) ─────────────────────── */

inline void kabsch_weighted(const double *P, const double *Q,
                            const uint8_t *valid, const double *w,
                            int N, double *R, double *t) {
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
        double I[9] = {1,0,0, 0,1,0, 0,0,1};
        std::memcpy(R, I, 72); t[0]=t[1]=t[2]=0; return;
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
    double q0=q[0],q1=q[1],q2=q[2],q3=q[3];
    R[0]=q0*q0+q1*q1-q2*q2-q3*q3; R[1]=2*(q1*q2+q0*q3); R[2]=2*(q1*q3-q0*q2);
    R[3]=2*(q1*q2-q0*q3); R[4]=q0*q0-q1*q1+q2*q2-q3*q3; R[5]=2*(q2*q3+q0*q1);
    R[6]=2*(q1*q3+q0*q2); R[7]=2*(q2*q3-q0*q1); R[8]=q0*q0-q1*q1-q2*q2+q3*q3;
    for (int d = 0; d < 3; d++)
        t[d] = cQ[d] - (cP[0]*R[0*3+d]+cP[1]*R[1*3+d]+cP[2]*R[2*3+d]);
}

/* ── Weighted TM refinement ──────────────────────────────────── */

inline void weighted_tm_refine(const double *pred, const double *native,
                               const uint8_t *valid, double d0, double Lnorm,
                               int N, int n_iters, double *R, double *t,
                               double *out_score) {
    double d0sq = std::max(d0*d0, 1e-12);
    double weights[4096];
    for (int iter = 0; iter < n_iters; iter++) {
        for (int i = 0; i < N; i++) {
            if (!valid[i]) { weights[i] = 0; continue; }
            double mx = pred[i*3]*R[0]+pred[i*3+1]*R[3]+pred[i*3+2]*R[6]+t[0];
            double my = pred[i*3]*R[1]+pred[i*3+1]*R[4]+pred[i*3+2]*R[7]+t[1];
            double mz = pred[i*3]*R[2]+pred[i*3+1]*R[5]+pred[i*3+2]*R[8]+t[2];
            double dx=mx-native[i*3], dy=my-native[i*3+1], dz=mz-native[i*3+2];
            double d2 = dx*dx+dy*dy+dz*dz;
            weights[i] = 1.0 / (1.0 + d2/d0sq);
        }
        kabsch_weighted(pred, native, valid, weights, N, R, t);
    }
    *out_score = tm_score_no_d8(pred, native, valid, R, t, d0, Lnorm, N);
}

#endif
