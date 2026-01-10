#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <math.h>
#include "gHits.cuh"
#include "gPlane.cuh"
#include "gTracklet.cuh"

#ifndef TRACKLET_FEATURES
#define TRACKLET_FEATURES 160
#endif
#ifndef MAX_TRACKLETS_PER_EVENT
#define MAX_TRACKLETS_PER_EVENT 128
#endif
#ifndef MAX_HITS_PER_EVENT
#define MAX_HITS_PER_EVENT 512
#endif
#ifndef GEOM_IS_MM
#define GEOM_IS_MM 1
#endif
#ifndef DRIFT_IS_MM
#define DRIFT_IS_MM 1
#endif
#ifndef PT_KICK_KMAG
  #ifdef E1039
    #define PT_KICK_KMAG (-0.3819216f)
  #else
    #define PT_KICK_KMAG (-0.4141f)
  #endif
#endif

__device__ __forceinline__ bool is_d0_x(int d) {
    return (d == 3 || d == 4);
}

__device__ __forceinline__ float* ptr3(float* base, int evt, int trk) {
    const size_t off = ((size_t)evt * MAX_TRACKLETS_PER_EVENT + trk) * TRACKLET_FEATURES;
    return base + off;
}

__device__ __forceinline__ const float* cptr3(const float* base, int evt, int trk) {
    const size_t off = ((size_t)evt * MAX_TRACKLETS_PER_EVENT + trk) * TRACKLET_FEATURES;
    return base + off;
}

static __device__ __forceinline__ float safe_sigma_x_cm(const gPlane& pl, int det) {
    float s = pl.resolution[det];
    if (DRIFT_IS_MM) {
        s *= 0.1f;
    }
    if (!(isfinite(s) && s > 0.003f && s < 0.5f)) {
        s = 0.03f;
    }
    return s;
}

static __device__ __forceinline__ bool x_from_spacing(const gPlane& pl, int det, int el, float& x_cm) {
    const float S = 0.1f;
    const float sp = S * pl.spacing[det];
    const float xo = S * pl.xoffset[det];
    const float ne = pl.nelem[det];

    if (!isfinite(sp) || !isfinite(xo) || sp == 0.f) {
        return false;
    }

    if (!(ne > 1.f && ne < 1e5f)) {
        x_cm = xo + (float(el) - 1.f) * sp;
    } else {
        x_cm = xo + (float(el) - 0.5f * (float(ne) - 1.f)) * sp;
    }
    return isfinite(x_cm);
}

static __device__ inline bool fit_line(const float* Z, const float* V, const float* W, int N, float& v0, float& tv, float& chi2) {
    if (N < 2) {
        return false;
    }

    double Sw = 0, Sz = 0, Szz = 0, Sv = 0, Szv = 0;

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        double w = W[i], z = Z[i], v = V[i];
        Sw += w;
        Sz += w * z;
        Szz += w * z * z;
        Sv += w * v;
        Szv += w * z * v;
    }

    const double D = Sw * Szz - Sz * Sz;
    if (!isfinite(D) || fabs(D) < 1e-12) {
        return false;
    }

    tv = (float)((Sw * Szv - Sz * Sv) / D);
    v0 = (float)((Szz * Sv - Sz * Szv) / D);

    double c2 = 0;
    for (int i = 0; i < N; ++i) {
        const double r = (double)V[i] - ((double)v0 + (double)tv * (double)Z[i]);
        c2 += (double)W[i] * r * r;
    }

    chi2 = (float)c2;
    return isfinite(v0) && isfinite(tv);
}

__global__ void gkernel_global_with_d0(const float* __restrict__ xz,
                                       const unsigned int* __restrict__ nxz,
                                       const float* __restrict__ yz,
                                       const unsigned int* __restrict__ nyz,
                                       const float* __restrict__ dHits,
                                       const int* __restrict__ dNHits,
                                       const float* __restrict__ dPlane,
                                       float* __restrict__ out,
                                       unsigned int* __restrict__ nout,
                                       int nEvents,
                                       float z_kmag_bend) {
    const int ie = blockIdx.x * blockDim.x + threadIdx.x;
    if (ie >= nEvents) {
        return;
    }

    for (int t = 0; t < 2; ++t) {
        float* dst = ptr3(out, ie, t);
        #pragma unroll
        for (int f = 0; f < TRACKLET_FEATURES; ++f) {
            dst[f] = NAN;
        }
    }

    const unsigned int NX = nxz[ie];
    const unsigned int NY = nyz[ie];
    int ibx = -1, iby = -1;
    float best = 1e30f;

    for (unsigned int i = 0; i < NX; ++i) {
        const float cx = cptr3(xz, ie, (int)i)[3];
        if (!isfinite(cx)) {
            continue;
        }

        if (NY == 0) {
            if (ibx < 0 || cx < cptr3(xz, ie, ibx)[3]) {
                ibx = (int)i;
            }
        } else {
            for (unsigned int j = 0; j < NY; ++j) {
                const float cy = cptr3(yz, ie, (int)j)[3];
                if (!isfinite(cy)) {
                    continue;
                }
                const float s = cx + cy;
                if (s < best) {
                    best = s;
                    ibx = (int)i;
                    iby = (int)j;
                }
            }
        }
    }

    unsigned int n = 0;
    if (ibx >= 0) {
        const float* src = cptr3(xz, ie, ibx);
        float* dst = ptr3(out, ie, (int)n);
        #pragma unroll
        for (int f = 0; f < TRACKLET_FEATURES; ++f) {
            dst[f] = src[f];
        }
        ++n;
    }
    if (iby >= 0) {
        const float* src = cptr3(yz, ie, iby);
        float* dst = ptr3(out, ie, (int)n);
        #pragma unroll
        for (int f = 0; f < TRACKLET_FEATURES; ++f) {
            dst[f] = src[f];
        }
        ++n;
    }
    nout[ie] = n;

    if (n < 1) {
        return;
    }

    ptr3(out, ie, 0)[14] = NAN;
    if (n >= 2) {
        ptr3(out, ie, 1)[14] = NAN;
    }

    const float tx_ds = ptr3(out, ie, 0)[5];
    const float x0_ds = ptr3(out, ie, 0)[7];
    if (!(isfinite(tx_ds) && isfinite(x0_ds))) {
        return;
    }

    const gPlane& pl = *reinterpret_cast<const gPlane*>(dPlane);
    gHits hits((float*)dHits + ie * MAX_HITS_PER_EVENT * 6, dNHits[ie]);
    const int nh = dNHits[ie];

    const int HMAX = 64;
    float X3[HMAX], Z3[HMAX], W3[HMAX]; int n3 = 0;
    float X4[HMAX], Z4[HMAX], W4[HMAX]; int n4 = 0;

    for (int h = 0; h < nh; ++h) {
        const int det = (int)hits.chan(h);
        if (!is_d0_x(det)) {
            continue;
        }
        const int el = (int)hits.pos(h);
        float xw = 0.f;
        if (!x_from_spacing(pl, det, el, xw)) {
            continue;
        }
        const float zh = pl.z[det];
        if (!isfinite(zh)) {
            continue;
        }
        const float w = 1.0f / (safe_sigma_x_cm(pl, det) * safe_sigma_x_cm(pl, det));
        if (det == 3 && n3 < HMAX) {
            X3[n3] = xw; Z3[n3] = zh; W3[n3] = w; ++n3;
        }
        if (det == 4 && n4 < HMAX) {
            X4[n4] = xw; Z4[n4] = zh; W4[n4] = w; ++n4;
        }
    }

    if (n3 == 0 || n4 == 0) {
        return;
    }

    const float x_ds_at_k = x0_ds + tx_ds * z_kmag_bend;
    float best_cost = 1e30f;
    float best_tx_st1 = NAN;

    for (int i = 0; i < n3; ++i) {
        for (int j = 0; j < n4; ++j) {
            const float dz = Z4[j] - Z3[i];
            if (!isfinite(dz) || fabsf(dz) < 1e-6f) {
                continue;
            }
            const float tx = (X4[j] - X3[i]) / dz;
            const float xk = X3[i] + tx * (z_kmag_bend - Z3[i]);
            const float cost = fabsf(xk - x_ds_at_k);
            if (cost < best_cost) {
                best_cost = cost;
                best_tx_st1 = tx;
            }
        }
    }

    if (!isfinite(best_tx_st1)) {
        return;
    }

    const float invP = fabsf((best_tx_st1 - tx_ds) / PT_KICK_KMAG);
    if (isfinite(invP)) {
        ptr3(out, ie, 0)[14] = invP;
        if (n >= 2) {
            ptr3(out, ie, 1)[14] = invP;
        }
    }
}

extern "C" void launch_gKernel_Global_Combine(void* dXZ,
                                              void* dNXZ,
                                              void* dYZ,
                                              void* dNYZ,
                                              void* dOut,
                                              void* dNOut,
                                              int nEvents,
                                              float z_ref_kmag_bend,
                                              void* dHits,
                                              void* dNHits,
                                              void* dPlane) {
    const int BS = 128;
    const dim3 grid((nEvents + BS - 1) / BS);
    gkernel_global_with_d0<<<grid, BS>>>((const float*)dXZ,
                                         (const unsigned int*)dNXZ,
                                         (const float*)dYZ,
                                         (const unsigned int*)dNYZ,
                                         (const float*)dHits,
                                         (const int*)dNHits,
                                         (const float*)dPlane,
                                         (float*)dOut,
                                         (unsigned int*)dNOut,
                                         nEvents,
                                         z_ref_kmag_bend);
    cudaDeviceSynchronize();
}
