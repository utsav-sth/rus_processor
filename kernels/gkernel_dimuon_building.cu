// kernels/gkernel_dimuon_building.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "gHits.cuh"
#include "gPlane.cuh"

#ifndef MAX_HITS_PER_EVENT
#define MAX_HITS_PER_EVENT 512
#endif
#ifndef MAX_TRACKLETS_PER_EVENT
#define MAX_TRACKLETS_PER_EVENT 128
#endif
#ifndef TRACKLET_FEATURES
#define TRACKLET_FEATURES 160
#endif

#define IDX_CHI2 3
#define IDX_T    5
#define IDX_X0   7

#ifndef DIMU_MAX_CANDS
#define DIMU_MAX_CANDS 16
#endif

#ifndef DIMU_MMU
#define DIMU_MMU 0.1056583745f
#endif
#ifndef DIMU_MP
#define DIMU_MP 0.9382720813f
#endif
#ifndef DIMU_PBEAM
#define DIMU_PBEAM 120.0f
#endif

static __device__ __forceinline__ bool isfin(float x) { return isfinite(x); }
static __device__ __forceinline__ float sqr(float x) { return x * x; }

static __device__ __forceinline__ const float* trk_ptr(const float* base, int evt, int idx) {
    return base + ((size_t)evt * (size_t)MAX_TRACKLETS_PER_EVENT + (size_t)idx) * (size_t)TRACKLET_FEATURES;
}

static __device__ __forceinline__ float mult4(const float* a, const float* b) {
    return a[3] * b[3] - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

static __device__ __forceinline__ float m2_4(const float* p) {
    return p[3] * p[3] - (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
}

static __device__ __forceinline__ float perp3(const float* p) {
    return sqrtf(p[0] * p[0] + p[1] * p[1]);
}

static __device__ __forceinline__ float energy(float m, float px, float py, float pz) {
    return sqrtf(m * m + px * px + py * py + pz * pz);
}

static __device__ __forceinline__ void boost_z(float* p4, float bz) {
    const float b2 = bz * bz;
    if (!(isfin(b2) && b2 < 1.0f)) {
        return;
    }
    const float gamma = rsqrtf(1.0f - b2);
    const float bp = p4[2] * bz;
    const float pz_new = p4[2] + (gamma - 1.0f) * bp * bz / (b2 > 0.0f ? b2 : 1.0f) + gamma * bz * p4[3];
    const float e_new  = gamma * (p4[3] + bp);
    p4[2] = pz_new;
    p4[3] = e_new;
}

static __device__ __forceinline__ bool x_from_spacing(const gPlane& pl, int det, int el, float& xcm) {
    const float ne = (float)pl.nelem[det];
    const float sp = pl.spacing[det];
    const float xo = pl.xoffset[det];
    if (!(isfin(ne) && ne > 0.0f && isfin(sp) && isfin(xo))) {
        return false;
    }
    // element assumed 1..N
    xcm = (((float)el - 0.5f) - 0.5f * ne) * sp + xo;
    return isfin(xcm);
}

struct Cand {
    float tx, x0;
    float ty, y0;
    float chi2;
    float invp;
    int   q;
    float px, py, pz;
};

// Build one track candidate
static __device__ __forceinline__ bool build_from_single_d0(
    const gPlane& pl,
    const gHits& hits,
    float tx_ds, float x0_ds,
    float ty_ds, float y0_ds,
    float z_kmag_bend,
    float pt_kick_kmag,
    float d0_win_cm,
    Cand& out)
{
    if (!(isfin(tx_ds) && isfin(x0_ds) && isfin(ty_ds) && isfin(y0_ds))) {
        return false;
    }
    if (!(isfin(z_kmag_bend) && isfin(pt_kick_kmag) && fabsf(pt_kick_kmag) > 1e-9f)) {
        return false;
    }

    int best_det = -1;
    int best_el  = -1;
    float best_x = NAN;
    float best_z = NAN;
    float best_res = 1e30f;

    const int nh = hits.NHitsTotal;
    #pragma unroll
    for (int h = 0; h < MAX_HITS_PER_EVENT; ++h) {
        if (h >= nh) break;
        const int det = (int)lrintf(hits.chan(h));
        if (det != 3 && det != 4) continue;
        const int el  = (int)lrintf(hits.pos(h));
        float xw;
        if (!x_from_spacing(pl, det, el, xw)) continue;
        const float z0 = pl.z[det];
        if (!isfin(z0)) continue;
        const float x_pred = x0_ds + tx_ds * z0;
        const float res = fabsf(x_pred - xw);
        if (res < best_res) {
            best_res = res;
            best_det = det;
            best_el  = el;
            best_x   = xw;
            best_z   = z0;
        }
    }

    if (best_det < 0 || !isfin(best_x) || !isfin(best_z)) {
        return false;
    }

    if (d0_win_cm > 0.0f && best_res > d0_win_cm) {
        return false;
    }

    const float dz = z_kmag_bend - best_z;
    if (!isfin(dz) || fabsf(dz) < 1e-6f) {
        return false;
    }

    // downstream position at kmag bend
    const float x_ds_at_k = x0_ds + tx_ds * z_kmag_bend;

    const float tx_st1 = (x_ds_at_k - best_x) / dz;

    const float invp = (tx_st1 - tx_ds) / pt_kick_kmag;
    if (!isfin(invp) || fabsf(invp) < 1e-9f) {
        return false;
    }

    const float p = 1.0f / fabsf(invp);
    const float norm = rsqrtf(1.0f + tx_ds * tx_ds + ty_ds * ty_ds);
    const float ux = tx_ds * norm;
    const float uy = ty_ds * norm;
    const float uz = 1.0f * norm;

    out.tx = tx_ds; out.x0 = x0_ds;
    out.ty = ty_ds; out.y0 = y0_ds;
    out.invp = invp;
    out.q = (invp > 0.0f) ? +1 : -1;
    out.px = p * ux;
    out.py = p * uy;
    out.pz = p * uz;
    return isfin(out.px) && isfin(out.py) && isfin(out.pz);
}

__global__ void gKernel_Dimuon_Building(
    const float* __restrict__ dXZ,
    const unsigned int* __restrict__ dNXZ,
    const float* __restrict__ dYZ,
    const unsigned int* __restrict__ dNYZ,
    const float* __restrict__ dHits,
    const int* __restrict__ dNHits,
    const float* __restrict__ dPlane,
    int* __restrict__ dNDimu,
    float* __restrict__ dDimVtx,   // [nEvents,3]
    float* __restrict__ dDimP,     // [nEvents,3]
    float* __restrict__ dDimMass,  // [nEvents]
    float* __restrict__ dDimChi2,  // [nEvents]
    float* __restrict__ dMu1PQ,    // [nEvents,4]
    float* __restrict__ dMu2PQ,    // [nEvents,4]
    float* __restrict__ dXF,       // [nEvents]
    float* __restrict__ dX1,
    float* __restrict__ dX2,
    float* __restrict__ dPT,
    float* __restrict__ dPhi,
    float* __restrict__ dCosTheta,
    int nEvents,
    float z_kmag_bend_cm,
    float pt_kick_kmag,
    float d0_win_cm,
    float mass_min,
    float mass_max,
    int require_opposite_charge)
{
    const int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;

    dNDimu[evt] = 0;

    dDimMass[evt] = NAN;
    dDimChi2[evt] = NAN;
    dXF[evt] = NAN;
    dX1[evt] = NAN;
    dX2[evt] = NAN;
    dPT[evt] = NAN;
    dPhi[evt] = NAN;
    dCosTheta[evt] = NAN;

    dDimVtx[3*evt + 0] = NAN;
    dDimVtx[3*evt + 1] = NAN;
    dDimVtx[3*evt + 2] = NAN;

    dDimP[3*evt + 0] = NAN;
    dDimP[3*evt + 1] = NAN;
    dDimP[3*evt + 2] = NAN;

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        dMu1PQ[4*evt + k] = NAN;
        dMu2PQ[4*evt + k] = NAN;
    }

    const unsigned int NX = dNXZ[evt];
    const unsigned int NY = dNYZ[evt];
    const unsigned int N  = (NX < NY) ? NX : NY;
    if (N < 2) return;

    const gPlane& pl = *reinterpret_cast<const gPlane*>(dPlane);
    const int nh = dNHits[evt];
    gHits hits((float*)dHits + (size_t)evt * (size_t)MAX_HITS_PER_EVENT * 6, nh);

    // Vertex/beam constants appended after gPlane (see utils/geometry.py)
    const int off_words = (int)(sizeof(gPlane) / sizeof(float));
    float ZU = dPlane[off_words + 2];
    float SX = dPlane[off_words + 4];
    float SY = dPlane[off_words + 6];
    if (!isfin(ZU)) ZU = -700.0f;
    if (!(isfin(SX) && SX > 0.f)) SX = 0.3f;
    if (!(isfin(SY) && SY > 0.f)) SY = 0.3f;

    // Select up to DIMU_MAX_CANDS best 3D track candidates by chi2 sum
    int cand_idx[DIMU_MAX_CANDS];
    float cand_score[DIMU_MAX_CANDS];
    int nC = 0;

    for (unsigned int i = 0; i < N; ++i) {
        const float* txz = trk_ptr(dXZ, evt, (int)i);
        const float* tyz = trk_ptr(dYZ, evt, (int)i);
        const float cx = txz[IDX_CHI2];
        const float cy = tyz[IDX_CHI2];
        const float tx = txz[IDX_T];
        const float x0 = txz[IDX_X0];
        const float ty = tyz[IDX_T];
        const float y0 = tyz[IDX_X0];
        if (!(isfin(cx) && isfin(cy) && isfin(tx) && isfin(x0) && isfin(ty) && isfin(y0))) {
            continue;
        }
        const float s = cx + cy;

        if (nC < DIMU_MAX_CANDS) {
            cand_idx[nC] = (int)i;
            cand_score[nC] = s;
            ++nC;
        } else {
            // replace worst
            int iw = 0;
            float sw = cand_score[0];
            for (int k = 1; k < DIMU_MAX_CANDS; ++k) {
                if (cand_score[k] > sw) { sw = cand_score[k]; iw = k; }
            }
            if (s < sw) {
                cand_idx[iw] = (int)i;
                cand_score[iw] = s;
            }
        }
    }

    if (nC < 2) return;

    Cand cands[DIMU_MAX_CANDS];
    int nV = 0;
    for (int k = 0; k < nC; ++k) {
        const int i = cand_idx[k];
        const float* txz = trk_ptr(dXZ, evt, i);
        const float* tyz = trk_ptr(dYZ, evt, i);
        Cand c;
        c.chi2 = txz[IDX_CHI2] + tyz[IDX_CHI2];
        const float tx = txz[IDX_T];
        const float x0 = txz[IDX_X0];
        const float ty = tyz[IDX_T];
        const float y0 = tyz[IDX_X0];
        if (!build_from_single_d0(pl, hits, tx, x0, ty, y0, z_kmag_bend_cm, pt_kick_kmag, d0_win_cm, c)) {
            continue;
        }
        cands[nV++] = c;
        if (nV >= DIMU_MAX_CANDS) break;
    }

    if (nV < 2) return;

    // Choose best pair by closest approach chi2
    float best_cost = 1e30f;
    int best_a = -1, best_b = -1;
    float best_z = NAN;

    for (int a = 0; a < nV; ++a) {
        for (int b = a + 1; b < nV; ++b) {
            if (require_opposite_charge) {
                if (cands[a].q * cands[b].q >= 0) continue;
            }
            const float dx0 = cands[a].x0 - cands[b].x0;
            const float dtx = cands[a].tx - cands[b].tx;
            const float dy0 = cands[a].y0 - cands[b].y0;
            const float dty = cands[a].ty - cands[b].ty;
            const float denom = dtx * dtx + dty * dty;
            if (!(isfin(denom) && denom > 1e-12f)) continue;
            float z = -(dx0 * dtx + dy0 * dty) / denom;
            if (!isfin(z)) continue;
            if (z > 300.f) z = 300.f;
            if (z < ZU) z = ZU;

            const float xa = cands[a].x0 + cands[a].tx * z;
            const float ya = cands[a].y0 + cands[a].ty * z;
            const float xb = cands[b].x0 + cands[b].tx * z;
            const float yb = cands[b].y0 + cands[b].ty * z;

            const float chi2 = sqr((xa - xb) / SX) + sqr((ya - yb) / SY);
            if (chi2 < best_cost) {
                best_cost = chi2;
                best_a = a;
                best_b = b;
                best_z = z;
            }
        }
    }

    if (best_a < 0 || best_b < 0 || !isfin(best_z)) return;

    // assign mu+ and mu- for kinematics
    Cand ca = cands[best_a];
    Cand cb = cands[best_b];
    Cand pos = ca;
    Cand neg = cb;
    if (ca.q < cb.q) { 
        pos = cb;
        neg = ca;
    } else if (ca.q == cb.q && ca.q < 0) {
        pos = cb;
        neg = ca;
    }

    // vertex
    const float x1 = ca.x0 + ca.tx * best_z;
    const float y1 = ca.y0 + ca.ty * best_z;
    const float x2 = cb.x0 + cb.tx * best_z;
    const float y2 = cb.y0 + cb.ty * best_z;
    const float vx = 0.5f * (x1 + x2);
    const float vy = 0.5f * (y1 + y2);
    const float vz = best_z;

    dDimVtx[3*evt + 0] = vx;
    dDimVtx[3*evt + 1] = vy;
    dDimVtx[3*evt + 2] = vz;

    // four-vectors (lab)
    float p_pos[4] = {pos.px, pos.py, pos.pz, 0.f};
    float p_neg[4] = {neg.px, neg.py, neg.pz, 0.f};
    p_pos[3] = energy(DIMU_MMU, p_pos[0], p_pos[1], p_pos[2]);
    p_neg[3] = energy(DIMU_MMU, p_neg[0], p_neg[1], p_neg[2]);

    float p_sum[4] = {
        p_pos[0] + p_neg[0],
        p_pos[1] + p_neg[1],
        p_pos[2] + p_neg[2],
        p_pos[3] + p_neg[3]
    };

    float m2 = m2_4(p_sum);
    float m = (m2 > 0.0f) ? sqrtf(m2) : 0.0f;

    if (!isfin(m) || m < mass_min || m > mass_max) {
        return;
    }

    dDimP[3*evt + 0] = p_sum[0];
    dDimP[3*evt + 1] = p_sum[1];
    dDimP[3*evt + 2] = p_sum[2];

    dDimMass[evt] = m;
    dDimChi2[evt] = best_cost;

    // per-muon
    dMu1PQ[4*evt + 0] = pos.px;
    dMu1PQ[4*evt + 1] = pos.py;
    dMu1PQ[4*evt + 2] = pos.pz;
    dMu1PQ[4*evt + 3] = (float)pos.q;

    dMu2PQ[4*evt + 0] = neg.px;
    dMu2PQ[4*evt + 1] = neg.py;
    dMu2PQ[4*evt + 2] = neg.pz;
    dMu2PQ[4*evt + 3] = (float)neg.q;

    // kinematics
    const float Ebeam = sqrtf(DIMU_PBEAM * DIMU_PBEAM + DIMU_MP * DIMU_MP);
    const float p_beam[4]   = {0.f, 0.f, DIMU_PBEAM, Ebeam};
    const float p_target[4] = {0.f, 0.f, 0.f, DIMU_MP};
    const float p_cms[4]    = {0.f, 0.f, DIMU_PBEAM, Ebeam + DIMU_MP};

    const float s = m2_4(p_cms);
    const float sqrt_s = (s > 0.0f) ? sqrtf(s) : NAN;

    // x1/x2
    const float denom_x1 = mult4(p_target, p_cms);
    const float denom_x2 = mult4(p_beam,   p_cms);
    if (isfin(denom_x1) && fabsf(denom_x1) > 1e-9f) {
        dX1[evt] = mult4(p_target, p_sum) / denom_x1;
    }
    if (isfin(denom_x2) && fabsf(denom_x2) > 1e-9f) {
        dX2[evt] = mult4(p_beam, p_sum) / denom_x2;
    }

    // pT
    const float pT = perp3(p_sum);
    dPT[evt] = pT;

    // xF (boost p_sum to CMS)
    if (isfin(sqrt_s) && sqrt_s > 0.0f) {
        float p_sum_cms[4] = {p_sum[0], p_sum[1], p_sum[2], p_sum[3]};
        const float beta = p_cms[2] / p_cms[3];
        boost_z(p_sum_cms, -beta);
        const float fac = (1.0f - (m * m) / s);
        if (isfin(fac) && fabsf(fac) > 1e-9f) {
            dXF[evt] = 2.0f * p_sum_cms[2] / (sqrt_s * fac);
        }
    }

    // Collins-Soper angles
    const float denom_cs = m * sqrtf(m * m + pT * pT);
    if (isfin(denom_cs) && denom_cs > 1e-9f) {
        dCosTheta[evt] = 2.0f * (p_neg[3] * p_pos[2] - p_neg[2] * p_pos[3]) / denom_cs;
        const float num_phi = 2.0f * sqrtf(m * m + pT * pT) * (p_neg[0] * p_pos[1] - p_neg[1] * p_pos[0]);
        const float den_phi = m * (p_pos[0] * p_pos[0] - p_neg[0] * p_neg[0] + p_pos[1] * p_pos[1] - p_neg[1] * p_neg[1]);
        dPhi[evt] = atan2f(num_phi, den_phi);
    }

    dNDimu[evt] = 1;
}

extern "C" void launch_gKernel_Dimuon_Building(
    void* dXZ,
    void* dNXZ,
    void* dYZ,
    void* dNYZ,
    void* dHits,
    void* dNHits,
    void* dPlane,
    void* dNDimu,
    void* dDimVtx,
    void* dDimP,
    void* dDimMass,
    void* dDimChi2,
    void* dMu1PQ,
    void* dMu2PQ,
    void* dXF,
    void* dX1,
    void* dX2,
    void* dPT,
    void* dPhi,
    void* dCosTheta,
    int nEvents,
    float z_kmag_bend_cm,
    float pt_kick_kmag,
    float d0_win_cm,
    float mass_min,
    float mass_max,
    int require_opposite_charge)
{
    const int threads = 128;
    const int blocks = (nEvents + threads - 1) / threads;
    gKernel_Dimuon_Building<<<blocks, threads>>>(
        (const float*)dXZ,
        (const unsigned int*)dNXZ,
        (const float*)dYZ,
        (const unsigned int*)dNYZ,
        (const float*)dHits,
        (const int*)dNHits,
        (const float*)dPlane,
        (int*)dNDimu,
        (float*)dDimVtx,
        (float*)dDimP,
        (float*)dDimMass,
        (float*)dDimChi2,
        (float*)dMu1PQ,
        (float*)dMu2PQ,
        (float*)dXF,
        (float*)dX1,
        (float*)dX2,
        (float*)dPT,
        (float*)dPhi,
        (float*)dCosTheta,
        nEvents,
        z_kmag_bend_cm,
        pt_kick_kmag,
        d0_win_cm,
        mass_min,
        mass_max,
        require_opposite_charge);
    cudaDeviceSynchronize();
}