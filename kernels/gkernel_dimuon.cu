#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <math.h>

#include "gHits.cuh"
#include "gPlane.cuh"

#ifndef TRACKLET_FEATURES
#define TRACKLET_FEATURES 160
#endif
#ifndef MAX_TRACKLETS_PER_EVENT
#define MAX_TRACKLETS_PER_EVENT 128
#endif
#ifndef MAX_HITS_PER_EVENT
#define MAX_HITS_PER_EVENT 512
#endif
#ifndef SPACING_IS_MM
#define SPACING_IS_MM 0
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

// Tracklet indices (must match your existing kernels/main.py usage)
#define IDX_CHI2 3
#define IDX_T    5
#define IDX_X0   7

static __device__ __forceinline__ bool finitef(float x) { return isfinite(x); }
static __device__ __forceinline__ float sqr(float x) { return x * x; }

__device__ __forceinline__ const float* cptr3(const float* base, int evt, int trk) {
  const size_t off = ((size_t)evt * MAX_TRACKLETS_PER_EVENT + (size_t)trk) * (size_t)TRACKLET_FEATURES;
  return base + off;
}

__device__ __forceinline__ bool is_d0_x(int d) {
  return (d == 3 || d == 4);
}

static __device__ __forceinline__ float safe_sigma_x_cm(const gPlane& pl, int det) {
  float s = pl.resolution[det];
  if (DRIFT_IS_MM) s *= 0.1f;
  if (!(isfinite(s) && s > 0.003f && s < 0.5f)) s = 0.03f;
  return s;
}

static __device__ __forceinline__ bool x_from_spacing(const gPlane& pl, int det, int el, float& x_cm) {
  const float S = (SPACING_IS_MM ? 0.1f : 1.0f);
  const float sp = S * pl.spacing[det];
  const float xo = S * pl.xoffset[det];
  const float ne = (float)pl.nelem[det];

  if (!isfinite(sp) || !isfinite(xo) || sp == 0.f) return false;

  if (!(ne > 1.f && ne < 1e5f)) {
    x_cm = xo + ((float)el - 1.f) * sp;
  } else {
    x_cm = xo + ((float)el - 0.5f * (ne - 1.f)) * sp;
  }
  return isfinite(x_cm);
}

static __device__ __forceinline__ bool compute_signed_invP_from_d0(
    const gPlane& pl,
    const gHits& hits,
    int nh,
    float z_kmag_bend,
    float d0_win_cm,
    float tx_ds, float x0_ds,
    float& invP_signed_out,
    float& best_cost_out)
{
  invP_signed_out = NAN;
  best_cost_out = NAN;

  const int HMAX = 64;
  float X3[HMAX], Z3[HMAX]; int n3 = 0;
  float X4[HMAX], Z4[HMAX]; int n4 = 0;

  for (int h = 0; h < nh; ++h) {
    const int det = (int)hits.chan(h);
    if (!is_d0_x(det)) continue;

    const int el = (int)hits.pos(h);

    float xw = 0.f;
    if (!x_from_spacing(pl, det, el, xw)) continue;

    const float zh = pl.z[det];
    if (!isfinite(zh)) continue;

    // keep arrays separated by plane detid
    if (det == 3 && n3 < HMAX) { X3[n3] = xw; Z3[n3] = zh; ++n3; }
    if (det == 4 && n4 < HMAX) { X4[n4] = xw; Z4[n4] = zh; ++n4; }
  }

  if (n3 == 0 || n4 == 0) return false;
  if (!(finitef(tx_ds) && finitef(x0_ds) && finitef(z_kmag_bend))) return false;

  const float x_ds_at_k = x0_ds + tx_ds * z_kmag_bend;

  float best_cost = 1e30f;
  float best_tx_st1 = NAN;

  for (int i = 0; i < n3; ++i) {
    for (int j = 0; j < n4; ++j) {
      const float dz = Z4[j] - Z3[i];
      if (!isfinite(dz) || fabsf(dz) < 1e-6f) continue;

      const float tx_st1 = (X4[j] - X3[i]) / dz;
      const float xk = X3[i] + tx_st1 * (z_kmag_bend - Z3[i]);
      const float cost = fabsf(xk - x_ds_at_k);

      if (cost < best_cost) {
        best_cost = cost;
        best_tx_st1 = tx_st1;
      }
    }
  }

  if (!isfinite(best_tx_st1)) return false;

  // enforce window cut (unlike your global kernel which always takes best)
  if (isfinite(d0_win_cm) && d0_win_cm > 0.f) {
    if (best_cost > d0_win_cm) return false;
  }

  const float invP_signed = (best_tx_st1 - tx_ds) / PT_KICK_KMAG;
  if (!isfinite(invP_signed) || fabsf(invP_signed) < 1e-9f) return false;

  invP_signed_out = invP_signed;
  best_cost_out = best_cost;
  return true;
}

__global__ void gKernel_Dimuon_Building(
    const float* __restrict__ xz,
    const unsigned int* __restrict__ nxz,
    const float* __restrict__ yz,
    const unsigned int* __restrict__ nyz,
    const float* __restrict__ dHits,
    const int* __restrict__ dNHits,
    const float* __restrict__ dPlane,
    int nEvents,
    float z_kmag_bend,
    float d0_win_cm,
    float* __restrict__ dimu_vtx,   // [nEvents, 3]
    float* __restrict__ dimu_p,     // [nEvents, 3]
    float* __restrict__ dimu_mass,  // [nEvents]
    float* __restrict__ dimu_chi2,  // [nEvents]
    float* __restrict__ mu1_pq,     // [nEvents, 4] px,py,pz,q
    float* __restrict__ mu2_pq      // [nEvents, 4]
)
{
  const int ie = blockIdx.x * blockDim.x + threadIdx.x;
  if (ie >= nEvents) return;

  // defaults
  dimu_vtx[3*ie + 0] = NAN; dimu_vtx[3*ie + 1] = NAN; dimu_vtx[3*ie + 2] = NAN;
  dimu_p[3*ie + 0] = NAN;   dimu_p[3*ie + 1] = NAN;   dimu_p[3*ie + 2] = NAN;
  dimu_mass[ie] = NAN;
  dimu_chi2[ie] = NAN;

  for (int k = 0; k < 4; ++k) {
    mu1_pq[4*ie + k] = NAN;
    mu2_pq[4*ie + k] = NAN;
  }

  const unsigned int NX = nxz[ie];
  const unsigned int NY = nyz[ie];
  if (NX == 0 || NY == 0) return;

  const unsigned int N = (NX < NY) ? NX : NY;

  // ---- select best candidates by chi2 (small) ----
  const int MAX_CANDS = 16;
  int cand_idx[MAX_CANDS];
  float cand_score[MAX_CANDS];
  int cand_n = 0;

  for (unsigned int i = 0; i < N; ++i) {
    const float* txz = cptr3(xz, ie, (int)i);
    const float* tyz = cptr3(yz, ie, (int)i);

    const float tx = txz[IDX_T];
    const float x0 = txz[IDX_X0];
    const float ty = tyz[IDX_T];
    const float y0 = tyz[IDX_X0];

    if (!(finitef(tx) && finitef(x0) && finitef(ty) && finitef(y0))) continue;

    float sx = txz[IDX_CHI2]; if (!finitef(sx)) sx = 1e6f;
    float sy = tyz[IDX_CHI2]; if (!finitef(sy)) sy = 1e6f;
    const float score = sx + sy;

    if (cand_n < MAX_CANDS) {
      cand_idx[cand_n] = (int)i;
      cand_score[cand_n] = score;
      ++cand_n;
    } else {
      // keep only if better than current worst
      int worst = 0;
      for (int k = 1; k < MAX_CANDS; ++k) if (cand_score[k] > cand_score[worst]) worst = k;
      if (score >= cand_score[worst]) continue;
      cand_idx[worst] = (int)i;
      cand_score[worst] = score;
    }
  }

  if (cand_n < 2) return;

  // ---- build per-candidate 3D tracks with signed invP from D0 ----
  const gPlane& pl = *reinterpret_cast<const gPlane*>(dPlane);
  gHits hits((float*)dHits + (size_t)ie * (size_t)MAX_HITS_PER_EVENT * 6u, dNHits[ie]);
  const int nh = dNHits[ie];

  float tx_c[MAX_CANDS], ty_c[MAX_CANDS], x0_c[MAX_CANDS], y0_c[MAX_CANDS];
  float px_c[MAX_CANDS], py_c[MAX_CANDS], pz_c[MAX_CANDS];
  int q_c[MAX_CANDS];
  int valid_n = 0;

  for (int ic = 0; ic < cand_n; ++ic) {
    const int it = cand_idx[ic];
    const float* txz = cptr3(xz, ie, it);
    const float* tyz = cptr3(yz, ie, it);

    const float tx = txz[IDX_T];
    const float x0 = txz[IDX_X0];
    const float ty = tyz[IDX_T];
    const float y0 = tyz[IDX_X0];

    float invP_signed = NAN;
    float best_cost = NAN;

    if (!compute_signed_invP_from_d0(pl, hits, nh, z_kmag_bend, d0_win_cm, tx, x0, invP_signed, best_cost)) {
      continue;
    }

    const float invPabs = fabsf(invP_signed);
    if (!(isfinite(invPabs) && invPabs > 1e-9f)) continue;

    const float p = 1.f / invPabs;

    const float norm = sqrtf(1.f + tx*tx + ty*ty);
    if (!(isfinite(norm) && norm > 0.f)) continue;

    const float uz = 1.f / norm;
    const float ux = tx * uz;
    const float uy = ty * uz;

    tx_c[valid_n] = tx;
    ty_c[valid_n] = ty;
    x0_c[valid_n] = x0;
    y0_c[valid_n] = y0;

    px_c[valid_n] = p * ux;
    py_c[valid_n] = p * uy;
    pz_c[valid_n] = p * uz;

    q_c[valid_n] = (invP_signed > 0.f) ? +1 : -1;
    ++valid_n;
  }

  if (valid_n < 2) return;

  // ---- choose best opposite-charge pair by closest approach in (x,y) vs z ----
  int best_a = -1, best_b = -1;
  float best_d2 = 1e30f;
  float best_z = NAN;

  for (int a = 0; a < valid_n; ++a) {
    for (int b = a + 1; b < valid_n; ++b) {
      if (q_c[a] * q_c[b] >= 0) continue;

      const float dx0 = x0_c[a] - x0_c[b];
      const float dy0 = y0_c[a] - y0_c[b];
      const float dtx = tx_c[a] - tx_c[b];
      const float dty = ty_c[a] - ty_c[b];

      const float denom = dtx*dtx + dty*dty;
      if (!(isfinite(denom) && denom > 1e-12f)) continue;

      float z = -(dx0*dtx + dy0*dty) / denom;
      // reasonable clamp (cm)
      if (z < -200.f) z = -200.f;
      if (z >  800.f) z =  800.f;

      const float dx = dx0 + dtx * z;
      const float dy = dy0 + dty * z;
      const float d2 = dx*dx + dy*dy;

      if (d2 < best_d2) {
        best_d2 = d2;
        best_a = a;
        best_b = b;
        best_z = z;
      }
    }
  }

  if (best_a < 0 || best_b < 0 || !isfinite(best_z)) return;

  // vertex point = mid-point of the two tracks evaluated at best_z
  const float x1 = x0_c[best_a] + tx_c[best_a] * best_z;
  const float y1 = y0_c[best_a] + ty_c[best_a] * best_z;
  const float x2 = x0_c[best_b] + tx_c[best_b] * best_z;
  const float y2 = y0_c[best_b] + ty_c[best_b] * best_z;

  const float vx = 0.5f * (x1 + x2);
  const float vy = 0.5f * (y1 + y2);
  const float vz = best_z;

  // dimuon momentum = sum
  const float px = px_c[best_a] + px_c[best_b];
  const float py = py_c[best_a] + py_c[best_b];
  const float pz = pz_c[best_a] + pz_c[best_b];

  // invariant mass using muon mass
  const float MU = 0.1056583745f;
  const float p1s = px_c[best_a]*px_c[best_a] + py_c[best_a]*py_c[best_a] + pz_c[best_a]*pz_c[best_a];
  const float p2s = px_c[best_b]*px_c[best_b] + py_c[best_b]*py_c[best_b] + pz_c[best_b]*pz_c[best_b];
  const float e1 = sqrtf(p1s + MU*MU);
  const float e2 = sqrtf(p2s + MU*MU);
  const float es = (e1 + e2);
  const float ps = (px*px + py*py + pz*pz);
  float m2 = es*es - ps;
  if (m2 < 0.f) m2 = 0.f;
  const float mass = sqrtf(m2);

  dimu_vtx[3*ie + 0] = vx;
  dimu_vtx[3*ie + 1] = vy;
  dimu_vtx[3*ie + 2] = vz;

  dimu_p[3*ie + 0] = px;
  dimu_p[3*ie + 1] = py;
  dimu_p[3*ie + 2] = pz;

  dimu_mass[ie] = mass;
  dimu_chi2[ie] = best_d2; // closest-approach distance^2 (cm^2)

  // mu1/mu2 outputs (keep ordering as found)
  mu1_pq[4*ie + 0] = px_c[best_a];
  mu1_pq[4*ie + 1] = py_c[best_a];
  mu1_pq[4*ie + 2] = pz_c[best_a];
  mu1_pq[4*ie + 3] = (float)q_c[best_a];

  mu2_pq[4*ie + 0] = px_c[best_b];
  mu2_pq[4*ie + 1] = py_c[best_b];
  mu2_pq[4*ie + 2] = pz_c[best_b];
  mu2_pq[4*ie + 3] = (float)q_c[best_b];
}

extern "C" void launch_gKernel_Dimuon_Building(
    void* dXZ,
    void* dNXZ,
    void* dYZ,
    void* dNYZ,
    void* dHits,
    void* dNHits,
    void* dPlane,
    int nEvents,
    float z_kmag_bend,
    float d0_win_cm,
    void* dDimuVtx,
    void* dDimuP,
    void* dDimuMass,
    void* dDimuChi2,
    void* dMu1PQ,
    void* dMu2PQ)
{
  const int BS = 128;
  const dim3 grid((nEvents + BS - 1) / BS);

  gKernel_Dimuon_Building<<<grid, BS>>>(
      (const float*)dXZ,
      (const unsigned int*)dNXZ,
      (const float*)dYZ,
      (const unsigned int*)dNYZ,
      (const float*)dHits,
      (const int*)dNHits,
      (const float*)dPlane,
      nEvents,
      z_kmag_bend,
      d0_win_cm,
      (float*)dDimuVtx,
      (float*)dDimuP,
      (float*)dDimuMass,
      (float*)dDimuChi2,
      (float*)dMu1PQ,
      (float*)dMu2PQ);

  cudaDeviceSynchronize();
}
