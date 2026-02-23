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
#ifndef DRIFT_IS_MM
#define DRIFT_IS_MM 1
#endif
#ifndef DEBUG_YZ
#define DEBUG_YZ 0
#endif
#define YZ_DEBUG_EVENTS_LIMIT 6
#ifndef TY_MAX
#define TY_MAX 0.10f
#endif
#ifndef Y0_MAX
#define Y0_MAX 50.0f
#endif
#ifndef REQUIRE_TWO_STATIONS
#define REQUIRE_TWO_STATIONS 0
#endif

static __device__ __forceinline__ bool is_uv(const int d) {
    return (d == 13 || d == 14 || d == 17 || d == 18 || d == 19 || d == 20 ||
            d == 23 || d == 24 || d == 25 || d == 26 || d == 29 || d == 30);
}

static __device__ __forceinline__ int station_of(int d) {
    if (d == 13 || d == 14) return 1;
    if (d >= 17 && d <= 20) return 2;
    if (d >= 23 && d <= 26) return 3;
    if (d == 29 || d == 30) return 4;
    return 0;
}

static __device__ __forceinline__ float safe_sigma_cm(const gPlane& pl, int det) {
    float s = pl.resolution[det];
    if (DRIFT_IS_MM) s *= 0.1f;
    if (!(isfinite(s) && s > 0.005f && s < 0.5f)) s = 0.03f;
    return s;
}

static __device__ __forceinline__ float sigmaY_for_det(const gPlane& plane, int det) {
    const float sig = safe_sigma_cm(plane, det);
    const float S = GEOM_IS_MM ? 0.1f : 1.0f;
    const float dpx = S * plane.deltapx[det];
    const float dpy = S * plane.deltapy[det];
    const float norm_xy = sqrtf(dpx * dpx + dpy * dpy);
    const float eps = 1e-6f;
    if (!(isfinite(norm_xy) && norm_xy > eps)) return sig * 10.0f;
    const float uY = fabsf(dpx) / norm_xy;
    if (uY < eps) return sig * 10.0f;
    return sig / uY;
}

static __device__ inline bool yz_from_uv_and_seed_LR(const gPlane& pl, int detid, int elid,
                                                     float x0, float tx, float drift_cm,
                                                     int hitsign, float& Ycm, float& Zcm) {
    const float S = GEOM_IS_MM ? 0.1f : 1.0f;

    const float p1x = S * (pl.p1x_w1[detid] + pl.dp1x[detid] * (elid - 1));
    const float p1y = S * (pl.p1y_w1[detid] + pl.dp1y[detid] * (elid - 1));
    const float p1z = S * (pl.p1z_w1[detid] + pl.dp1z[detid] * (elid - 1));

    const float dpx = S * pl.deltapx[detid];
    const float dpy = S * pl.deltapy[detid];
    const float dpz = S * pl.deltapz[detid];

    const float z0 = pl.z[detid];

    const float denom = (dpx - tx * dpz);
    if (!isfinite(denom) || fabsf(denom) < 1e-10f) return false;

    const float t = (x0 + tx * (z0 + p1z) - p1x) / denom;
    float Yabs = p1y + t * dpy;
    const float Zabs = z0 + (p1z + t * dpz);

    if (!(isfinite(Yabs) && isfinite(Zabs))) return false;

    if (hitsign != 0 && isfinite(drift_cm) && drift_cm > 0.f) {
        const float norm_xy = sqrtf(dpx * dpx + dpy * dpy);
        const float eps = 1e-6f;
        if (isfinite(norm_xy) && norm_xy > eps) {
            const float nY_signed = dpx / norm_xy;
            Yabs += hitsign * drift_cm * nY_signed;
        } else {
            Yabs += hitsign * drift_cm;
        }
    }

    Ycm = Yabs;
    Zcm = Zabs;
    return true;
}

static __device__ inline bool wls_fit(const float* Z, const float* Y, const float* W, int N,
                                      float& y0, float& ty, float& ey0, float& ety, float& chi2) {
    if (N < 2) return false;

    double Sw = 0, Szw = 0, Szzw = 0, Syw = 0, Szyw = 0;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        const double w = W[i], z = Z[i], y = Y[i];
        Sw  += w;
        Szw += w * z;
        Szzw += w * z * z;
        Syw += w * y;
        Szyw += w * z * y;
    }

    const double D = Sw * Szzw - Szw * Szw;
    if (!(isfinite(D) && fabs(D) > 1e-12)) return false;

    ty  = (float)((Sw * Szyw - Szw * Syw) / D);
    y0  = (float)((Szzw * Syw - Szw * Szyw) / D);
    ety = (float)sqrt(fabs(Sw / D));
    ey0 = (float)sqrt(fabs(Szzw / D));

    double c2 = 0;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        const double r = (double)Y[i] - ((double)y0 + (double)ty * (double)Z[i]);
        c2 += (double)W[i] * r * r;
    }
    chi2 = (float)c2;
    return isfinite(y0) && isfinite(ty);
}

__global__ void gKernel_YZ_tracking(const float* __restrict__ dHits,
                                    const int* __restrict__ dNHits,
                                    const float* __restrict__ dPlane,
                                    const float* __restrict__ dXZ,
                                    const unsigned int* __restrict__ nXZ,
                                    float* __restrict__ dYZ_out,
                                    unsigned int* __restrict__ nYZ_out,
                                    const bool* __restrict__ HasTooManyHits,
                                    const int nEvents) {
    const int evt = blockIdx.x;
    if (evt >= nEvents) return;

    if (HasTooManyHits && HasTooManyHits[evt]) {
        nYZ_out[evt] = 0;
        return;
    }

    const gPlane& plane = *reinterpret_cast<const gPlane*>(dPlane);
    const int nh = dNHits[evt];
    gHits hits(const_cast<float*>(dHits) + evt * MAX_HITS_PER_EVENT * 6, nh);

    const unsigned int NX = nXZ[evt];
    const unsigned int NXcap = (NX < (unsigned)MAX_TRACKLETS_PER_EVENT) ? NX : (unsigned)MAX_TRACKLETS_PER_EVENT;
    constexpr int STRIDE = 160;

    // init outputs [0..NXcap-1] to NAN
    for (unsigned int i = 0; i < NXcap; ++i) {
        float* dst = (float*)dYZ_out + ((size_t)evt * MAX_TRACKLETS_PER_EVENT + i) * STRIDE;
        #pragma unroll
        for (int f = 0; f < STRIDE; ++f) dst[f] = NAN;
    }

    unsigned int kept = 0;

    for (unsigned int i = 0; i < NXcap; ++i) {
        const float* seed = (const float*)dXZ + ((size_t)evt * MAX_TRACKLETS_PER_EVENT + i) * STRIDE;
        const float tx = seed[5];
        const float x0 = seed[7];
        if (!isfinite(tx) || !isfinite(x0)) continue;

        const int HMAX = 128;
        float Z0[HMAX], Y0[HMAX], W0[HMAX];
        int n0 = 0;
        int sHit[5] = {0, 0, 0, 0, 0};

        for (int h = 0; h < nh && n0 < HMAX; ++h) {
            const int det = (int)hits.chan(h);
            if (!is_uv(det)) continue;

            const int el = (int)hits.pos(h);
            float y, z;
            if (!yz_from_uv_and_seed_LR(plane, det, el, x0, tx, 0.f, 0, y, z)) continue;

            const float sY = sigmaY_for_det(plane, det);
            const float w = 1.f / (sY * sY);
            Z0[n0] = z;
            Y0[n0] = y;
            W0[n0] = w;
            ++n0;

            int st = station_of(det);
            if (st >= 1 && st <= 4) sHit[st]++;
        }

        if (n0 < 2) continue;

        #if REQUIRE_TWO_STATIONS
        int nStations = (sHit[1] > 0) + (sHit[2] > 0) + (sHit[3] > 0) + (sHit[4] > 0);
        if (nStations < 2) continue;
        #endif

        float y0, ty, ey0, ety, chi2;
        if (!wls_fit(Z0, Y0, W0, n0, y0, ty, ey0, ety, chi2)) continue;

        float Z[HMAX], Y[HMAX], W[HMAX];
        int n = 0;

        for (int h = 0; h < nh && n < HMAX; ++h) {
            const int det = (int)hits.chan(h);
            if (!is_uv(det)) continue;

            const int el = (int)hits.pos(h);
            float drift_cm = hits.drift(h);
            if (DRIFT_IS_MM) drift_cm *= 0.1f;

            float yL = 0, zL = 0, yR = 0, zR = 0;
            bool okL = yz_from_uv_and_seed_LR(plane, det, el, x0, tx, drift_cm, -1, yL, zL);
            bool okR = yz_from_uv_and_seed_LR(plane, det, el, x0, tx, drift_cm, +1, yR, zR);
            if (!okL && !okR) continue;

            float ypick, zpick;
            if (okL && okR) {
                const float rL = fabsf(yL - (y0 + ty * zL));
                const float rR = fabsf(yR - (y0 + ty * zR));
                if (rL <= rR) { ypick = yL; zpick = zL; }
                else          { ypick = yR; zpick = zR; }
            } else if (okL) {
                ypick = yL; zpick = zL;
            } else {
                ypick = yR; zpick = zR;
            }

            const float sY = sigmaY_for_det(plane, det);
            const float w = 1.f / (sY * sY);
            Z[n] = zpick;
            Y[n] = ypick;
            W[n] = w;
            ++n;
        }

        if (n < 2) continue;
        if (!wls_fit(Z, Y, W, n, y0, ty, ey0, ety, chi2)) continue;

        if (!(isfinite(y0) && isfinite(ty)) || fabsf(y0) > Y0_MAX || fabsf(ty) > TY_MAX) continue;

        float* out = (float*)dYZ_out + ((size_t)evt * MAX_TRACKLETS_PER_EVENT + i) * STRIDE;
        out[0] = 1;
        out[1] = evt;
        out[2] = n;
        out[3] = chi2;
        out[5] = ty;
        out[7] = y0;
        out[10] = ety;
        out[12] = ey0;

        ++kept;
    }

    // This kernel does NOT compact outputs; it writes into slot i (seed index).
    // Returning kept would hide valid solutions at higher i.
    nYZ_out[evt] = NXcap;
}

extern "C" void launch_gKernel_YZ_tracking(float* dHits,
                                           int* dNHits,
                                           float* dPlane,
                                           float* dXZ,
                                           unsigned int* dNXZ,
                                           float* dYZ,
                                           unsigned int* dNYZ,
                                           bool* dHasTooMany,
                                           int nEvents) {
    dim3 block(1), grid(nEvents);
    gKernel_YZ_tracking<<<grid, block>>>(dHits, dNHits, dPlane, dXZ, dNXZ, dYZ, dNYZ, dHasTooMany, nEvents);
    cudaDeviceSynchronize();
}