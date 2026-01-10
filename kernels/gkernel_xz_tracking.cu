#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include "gHits.cuh"
#include "gPlane.cuh"
#include "gTracklet.cuh"

#ifndef MAX_TRACKLETS_PER_EVENT
#define MAX_TRACKLETS_PER_EVENT 128
#endif
#ifndef MAX_HITS_PER_EVENT
#define MAX_HITS_PER_EVENT 512
#endif
#ifndef GEOM_IS_MM
#define GEOM_IS_MM 1
#endif
#ifndef SPACING_IS_MM
#define SPACING_IS_MM 0
#endif
#ifndef DRIFT_IS_MM
#define DRIFT_IS_MM 1
#endif
#ifndef PAIR_WIN_FACTOR
#define PAIR_WIN_FACTOR 1.60f
#endif
#ifndef X0_MAX
#define X0_MAX 500.0f
#endif
#ifndef TX_MAX
#define TX_MAX 1.0f
#endif
#ifndef DEBUG_XZ
#define DEBUG_XZ 0
#endif

static __device__ __forceinline__ bool is_x_det(const int d) {
    return (d == 15 || d == 16 || d == 21 || d == 22 || d == 27 || d == 28);
}
static __device__ __forceinline__ int station_of(const int d) {
    if (d == 15 || d == 16) {
        return 1;
    }
    if (d == 21 || d == 22) {
        return 2;
    }
    if (d == 27 || d == 28) {
        return 3;
    }
    return 0;
}
static __device__ __forceinline__ bool is_A(const int d) {
    return (d == 15 || d == 21 || d == 27);
}
static __device__ __forceinline__ bool is_B(const int d) {
    return (d == 16 || d == 22 || d == 28);
}

static __device__ __forceinline__ bool x_from_spacing(const gPlane& pl, int det, int el, float& x_cm) {
    const float Ssp = (SPACING_IS_MM ? 0.1f : 1.0f);
    const float sp = Ssp * pl.spacing[det];
    const float xo = Ssp * pl.xoffset[det];
    const float ne = pl.nelem[det];
    if (!isfinite(sp) || !isfinite(xo) || sp == 0.0f) {
        return false;
    }
    if (!(ne > 1.f && ne < 1e5f)) {
        x_cm = xo + (float(el) - 1.f) * sp;
        return isfinite(x_cm);
    }
    x_cm = xo + (float(el) - 0.5f * (float(ne) - 1.f)) * sp;
    return isfinite(x_cm);
}

static __device__ __forceinline__ float safe_sigma_x_cm(const gPlane& pl, int det) {
    float s = pl.resolution[det];
    if (DRIFT_IS_MM) {
        s *= 0.1f;
    }
    if (!isfinite(s) || s < 0.01f || s > 0.5f) {
        s = 0.03f;
    }
    return s;
}

struct Pair {
    float zA, xA, zB, xB, wA, wB;
    int detA, detB, elA, elB;
};

static __device__ __forceinline__ int make_pairs_station(const gHits& hits, const gPlane& pl, int st, Pair* out, int maxN) {
    const int detA = (st == 1 ? 15 : (st == 2 ? 21 : 27));
    const int detB = (st == 1 ? 16 : (st == 2 ? 22 : 28));
    const float zA = pl.z[detA];
    const float zB = pl.z[detB];

    float xA[64];
    int eA[64];
    int nA = 0;
    float xB[64];
    int eB[64];
    int nB = 0;

    for (int i = 0; i < hits.NHitsTotal; ++i) {
        const int det = (int)hits.chan(i);
        if (det != detA && det != detB) {
            continue;
        }
        const int el = (int)hits.pos(i);
        float x = 0.f;
        if (!x_from_spacing(pl, det, el, x)) {
            continue;
        }
        if (det == detA && nA < 64) {
            xA[nA] = x;
            eA[nA] = el;
            ++nA;
        }
        if (det == detB && nB < 64) {
            xB[nB] = x;
            eB[nB] = el;
            ++nB;
        }
    }

    const float Ssp = (SPACING_IS_MM ? 0.1f : 1.0f);
    const float spacing_cm = Ssp * pl.spacing[detA];
    const float win = fmaxf(0.5f * spacing_cm, PAIR_WIN_FACTOR * spacing_cm);

    int np = 0;
    for (int i = 0; i < nA && np < maxN; ++i) {
        for (int j = 0; j < nB && np < maxN; ++j) {
            if (fabsf(xA[i] - xB[j]) <= win) {
                Pair p;
                p.zA = zA;
                p.zB = zB;
                p.xA = xA[i];
                p.xB = xB[j];
                p.detA = detA;
                p.detB = detB;
                p.elA = eA[i];
                p.elB = eB[j];
                const float sA = safe_sigma_x_cm(pl, detA);
                const float sB = safe_sigma_x_cm(pl, detB);
                p.wA = 1.f / (sA * sA);
                p.wB = 1.f / (sB * sB);
                out[np++] = p;
            }
        }
    }
    return np;
}

static __device__ __forceinline__ bool fit_wls(const float* Z, const float* X, const float* W, int N, float& x0, float& tx, float& ex0, float& etx, float& chi2) {
    if (N < 2) {
        return false;
    }
    double Sw = 0, Szw = 0, Szzw = 0, Sxw = 0, Szxw = 0;
    for (int i = 0; i < N; ++i) {
        const double w = W[i];
        const double z = Z[i];
        const double x = X[i];
        Sw += w;
        Szw += w * z;
        Szzw += w * z * z;
        Sxw += w * x;
        Szxw += w * z * x;
    }
    const double D = Sw * Szzw - Szw * Szw;
    if (!(isfinite(D) && fabs(D) > 1e-12)) {
        return false;
    }
    tx = (float)((Sw * Szxw - Szw * Sxw) / D);
    x0 = (float)((Szzw * Sxw - Szw * Szxw) / D);
    etx = (float)sqrt(fabs(Sw / D));
    ex0 = (float)sqrt(fabs(Szzw / D));
    double c2 = 0;
    for (int i = 0; i < N; ++i) {
        const double r = (double)X[i] - ((double)x0 + (double)tx * (double)Z[i]);
        c2 += (double)W[i] * r * r;
    }
    chi2 = (float)c2;
    return isfinite(x0) && isfinite(tx);
}

static __device__ __forceinline__ bool too_close(float x0a, float txa, float x0b, float txb) {
    return (fabsf(x0a - x0b) < 0.2f && fabsf(txa - txb) < 0.0005f);
}

__global__ void gKernel_XZ_tracking(const float* __restrict__ dHits,
                                    const int* __restrict__ dNHits,
                                    const float* __restrict__ dPlane,
                                    float* __restrict__ tracklet_output,
                                    unsigned int* __restrict__ nTracklets,
                                    const bool* __restrict__ HasTooManyHits,
                                    const int nEvents) {
    const int evt = blockIdx.x;
    if (evt >= nEvents) {
        return;
    }
    nTracklets[evt] = 0;
    if (HasTooManyHits && HasTooManyHits[evt]) {
        return;
    }

    const gPlane& plane = *reinterpret_cast<const gPlane*>(dPlane);
    const int nh = dNHits[evt];
    gHits hits(const_cast<float*>(dHits) + evt * MAX_HITS_PER_EVENT * 6, nh);

    Pair p1[64], p2[64], p3[64];
    const int n1 = make_pairs_station(hits, plane, 1, p1, 64);
    const int n2 = make_pairs_station(hits, plane, 2, p2, 64);
    const int n3 = make_pairs_station(hits, plane, 3, p3, 64);

    constexpr int STRIDE = 160;
    gTracklet tko(tracklet_output, evt * MAX_TRACKLETS_PER_EVENT * STRIDE);

    int nOut = 0;
    auto emit = [&](float x0, float tx, float ex0, float etx, float chi2, int Npts) {
        for (int k = 0; k < nOut; ++k) {
            float* prev = tko.m_trackletdata + k * STRIDE;
            if (too_close(prev[7], prev[5], x0, tx)) {
                return;
            }
        }
        if (!(isfinite(x0) && isfinite(tx))) {
            return;
        }
        if (fabsf(x0) > X0_MAX || fabsf(tx) > TX_MAX) {
            return;
        }
        if (nOut >= MAX_TRACKLETS_PER_EVENT) {
            return;
        }
        float* base = tko.m_trackletdata + nOut * STRIDE;
        base[0] = 1;
        base[1] = evt;
        base[2] = Npts;
        base[3] = chi2;
        base[5] = tx;
        base[7] = x0;
        base[10] = etx;
        base[12] = ex0;
        ++nOut;
    };

    auto fit_from = [&](float zA, float xA, float wA,
                         float zB, float xB, float wB,
                         float zC, float xC, float wC,
                         int Npts) -> void {
        float Z[3], X[3], W[3];
        Z[0] = zA; X[0] = xA; W[0] = wA;
        Z[1] = zB; X[1] = xB; W[1] = wB;
        if (Npts == 3) {
            Z[2] = zC; X[2] = xC; W[2] = wC;
        }
        float x0, tx, ex0, etx, chi2;
        if (fit_wls(Z, X, W, Npts, x0, tx, ex0, etx, chi2)) {
            emit(x0, tx, ex0, etx, chi2, Npts);
        }
    };

    for (int i = 0; i < n1 && nOut < MAX_TRACKLETS_PER_EVENT; ++i) {
        const float z1 = 0.5f * (p1[i].zA + p1[i].zB);
        const float x1 = 0.5f * (p1[i].xA + p1[i].xB);
        const float w1 = 0.5f * (p1[i].wA + p1[i].wB);
        for (int j = 0; j < n3 && nOut < MAX_TRACKLETS_PER_EVENT; ++j) {
            const float z3 = 0.5f * (p3[j].zA + p3[j].zB);
            const float x3 = 0.5f * (p3[j].xA + p3[j].xB);
            const float w3 = 0.5f * (p3[j].wA + p3[j].wB);
            if (n2 > 0) {
                float bestdz = 1e30f, z2b = 0, x2b = 0, w2b = 1.0f;
                bool have = false;
                const float tx_est = (x3 - x1) / (z3 - z1 + 1e-9f);
                const float x02_est = x1 - tx_est * z1;
                for (int k = 0; k < n2; ++k) {
                    const float z2 = 0.5f * (p2[k].zA + p2[k].zB);
                    const float x2 = 0.5f * (p2[k].xA + p2[k].xB);
                    const float w2 = 0.5f * (p2[k].wA + p2[k].wB);
                    const float xpred = x02_est + tx_est * z2;
                    const float dz = fabsf(x2 - xpred);
                    if (dz < bestdz) {
                        bestdz = dz;
                        z2b = z2;
                        x2b = x2;
                        w2b = w2;
                        have = true;
                    }
                }
                if (have) {
                    fit_from(z1, x1, w1, z3, x3, w3, z2b, x2b, w2b, 3);
                } else {
                    fit_from(z1, x1, w1, z3, x3, w3, 0, 0, 0, 2);
                }
            } else {
                fit_from(z1, x1, w1, z3, x3, w3, 0, 0, 0, 2);
            }
        }
    }

    if (nOut == 0) {
        for (int i = 0; i < n1 && nOut < MAX_TRACKLETS_PER_EVENT; ++i) {
            const float z1 = 0.5f * (p1[i].zA + p1[i].zB);
            const float x1 = 0.5f * (p1[i].xA + p1[i].xB);
            const float w1 = 0.5f * (p1[i].wA + p1[i].wB);
            for (int j = 0; j < n2 && nOut < MAX_TRACKLETS_PER_EVENT; ++j) {
                const float z2 = 0.5f * (p2[j].zA + p2[j].zB);
                const float x2 = 0.5f * (p2[j].xA + p2[j].xB);
                const float w2 = 0.5f * (p2[j].wA + p2[j].wB);
                fit_from(z1, x1, w1, z2, x2, w2, 0, 0, 0, 2);
            }
        }
        for (int i = 0; i < n2 && nOut < MAX_TRACKLETS_PER_EVENT; ++i) {
            const float z2 = 0.5f * (p2[i].zA + p2[i].zB);
            const float x2 = 0.5f * (p2[i].xA + p2[i].xB);
            const float w2 = 0.5f * (p2[i].wA + p2[i].wB);
            for (int j = 0; j < n3 && nOut < MAX_TRACKLETS_PER_EVENT; ++j) {
                const float z3 = 0.5f * (p3[j].zA + p3[j].zB);
                const float x3 = 0.5f * (p3[j].xA + p3[j].xB);
                const float w3 = 0.5f * (p3[j].wA + p3[j].wB);
                fit_from(z2, x2, w2, z3, x3, w3, 0, 0, 0, 2);
            }
        }
    }

#if DEBUG_XZ
    if (threadIdx.x == 0 && evt < 10) {
        for (int k = 0; k < nOut && k < 4; ++k) {
            float* base = tko.m_trackletdata + k * STRIDE;
            printf("[XZ-FIT] evt=%d k=%d tx=% .5f x0=% .5f chi2=%.3g N=%g\n", evt, k, base[5], base[7], base[3], base[2]);
        }
    }
#endif
    nTracklets[evt] = nOut;
}

extern "C" void launch_gKernel_XZ_tracking(float* dHits,
                                           int* dNHits,
                                           float* dTracklets,
                                           float* dPlane,
                                           unsigned int* dNTracklets,
                                           bool* dHasTooMany,
                                           int nEvents) {
    dim3 block(1), grid(nEvents);
    gKernel_XZ_tracking<<<grid, block>>>(dHits, dNHits, dPlane, dTracklets, dNTracklets, dHasTooMany, nEvents);
    cudaDeviceSynchronize();
}
