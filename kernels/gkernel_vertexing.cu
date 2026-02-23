#include <cuda_runtime.h>
#include <cuda.h>

#include <math.h>
#include <stdint.h>

#include "gPlane.cuh"

#ifndef TRACKLET_FEATURES
#define TRACKLET_FEATURES 160
#endif
#ifndef MAX_TRACKLETS_PER_EVENT
#define MAX_TRACKLETS_PER_EVENT 128
#endif

// gkernel_vertexing.cu

__device__ __forceinline__ const float* cptr3(const float* base, int evt, int trk) {
  const size_t off = ((size_t)evt * MAX_TRACKLETS_PER_EVENT + (size_t)trk) * (size_t)TRACKLET_FEATURES;
  return base + off;
}

__device__ __forceinline__ float sqr(float x) { return x * x; }

__device__ __forceinline__ bool finite_pos(float x) {
  return isfinite(x) && x > 0.f;
}

__device__ __forceinline__ float chi2_xy_at_z(
    float z,
    bool has_xz, float x0, float tx,
    bool has_yz, float y0, float ty,
    float xb, float sx,
    float yb, float sy)
{
  float c2 = 0.f;
  if (has_xz) {
    const float x = x0 + tx * z;
    c2 += sqr((x - xb) / sx);
  }
  if (has_yz) {
    const float y = y0 + ty * z;
    c2 += sqr((y - yb) / sy);
  }
  return c2;
}

__global__ void gKernel_Vertexing(
    const float* __restrict__ tracks,
    const unsigned int* __restrict__ ntracks,
    const float* __restrict__ dPlane,
    int nEvents,
    float* __restrict__ vtx_out,   // [nEvents, 3] = vx,vy,vz
    float* __restrict__ mom_out,   // [nEvents, 3] = px,py,pz
    float* __restrict__ chi2_out   // [nEvents, 3] = chi2_up, chi2_target, chi2_dump (sum x+y terms)
)
{
  const int evt = blockIdx.x * blockDim.x + threadIdx.x;
  if (evt >= nEvents) return;

  // Defaults
  vtx_out[3 * evt + 0] = NAN;
  vtx_out[3 * evt + 1] = NAN;
  vtx_out[3 * evt + 2] = NAN;
  mom_out[3 * evt + 0] = NAN;
  mom_out[3 * evt + 1] = NAN;
  mom_out[3 * evt + 2] = NAN;
  chi2_out[3 * evt + 0] = NAN;
  chi2_out[3 * evt + 1] = NAN;
  chi2_out[3 * evt + 2] = NAN;

  const unsigned int nt = ntracks[evt];
  if (nt == 0) return;

  // Read XZ (slot 0)
  bool has_xz = false;
  float tx = NAN, x0 = NAN;
  {
    const float* t0 = cptr3(tracks, evt, 0);
    tx = t0[5];
    x0 = t0[7];
    has_xz = isfinite(tx) && isfinite(x0);
  }

  // Read YZ (slot 1), if exists
  bool has_yz = false;
  float ty = 0.f, y0 = 0.f;
  if (nt >= 2) {
    const float* t1 = cptr3(tracks, evt, 1);
    ty = t1[5];
    y0 = t1[7];
    has_yz = isfinite(ty) && isfinite(y0);
  }

  if (!has_xz && !has_yz) return;

  float invP = NAN;
  {
    const float* t0 = cptr3(tracks, evt, 0);
    invP = t0[14];
    if (!isfinite(invP) && nt >= 2) {
      const float* t1 = cptr3(tracks, evt, 1);
      invP = t1[14];
    }
  }

  // Read vertex/beam constants appended after gPlane
  const int off_words = (int)(sizeof(gPlane) / sizeof(float));
  const float ZT = dPlane[off_words + 0];
  const float ZD = dPlane[off_words + 1];
  const float ZU = dPlane[off_words + 2];
  const float XB = dPlane[off_words + 3];
  float SX = dPlane[off_words + 4];
  const float YB = dPlane[off_words + 5];
  float SY = dPlane[off_words + 6];

  if (!finite_pos(SX)) SX = 0.2f;
  if (!finite_pos(SY)) SY = 0.2f;

  chi2_out[3 * evt + 0] = chi2_xy_at_z(ZU, has_xz, x0, tx, has_yz, y0, ty, XB, SX, YB, SY);
  chi2_out[3 * evt + 1] = chi2_xy_at_z(ZT, has_xz, x0, tx, has_yz, y0, ty, XB, SX, YB, SY);
  chi2_out[3 * evt + 2] = chi2_xy_at_z(ZD, has_xz, x0, tx, has_yz, y0, ty, XB, SX, YB, SY);

  const float step[3] = {25.f, 5.f, 1.f};
  float z_min = 300.f;
  float chi2_min = 1e30f;

  for (int i = 0; i < 3; ++i) {
    float z_start, z_end;
    if (i == 0) {
      z_start = z_min;
      z_end   = ZU;
    } else {
      z_start = z_min + step[i - 1];
      z_end   = z_min - step[i - 1];
    }

    // Ensure we scan decreasing z as in ktracker code
    for (float z = z_start; z > z_end; z -= step[i]) {
      const float c2 = chi2_xy_at_z(z, has_xz, x0, tx, has_yz, y0, ty, XB, SX, YB, SY);
      if (c2 < chi2_min) {
        chi2_min = c2;
        z_min = z;
      }
    }
  }

  // Evaluate vertex x,y at z_min
  float vx = XB;
  float vy = YB;
  if (has_xz) vx = x0 + tx * z_min;
  if (has_yz) vy = y0 + ty * z_min;

  vtx_out[3 * evt + 0] = vx;
  vtx_out[3 * evt + 1] = vy;
  vtx_out[3 * evt + 2] = z_min;

  // Momentum at vertex (straight-line approximation)
  if (isfinite(invP) && fabsf(invP) > 1e-9f) {
    const float p = 1.f / fabsf(invP);
    const float txx = has_xz ? tx : 0.f;
    const float tyy = has_yz ? ty : 0.f;
    const float norm = sqrtf(1.f + txx * txx + tyy * tyy);
    if (isfinite(norm) && norm > 0.f) {
      const float uz = 1.f / norm;
      const float ux = txx * uz;
      const float uy = tyy * uz;
      mom_out[3 * evt + 0] = p * ux;
      mom_out[3 * evt + 1] = p * uy;
      mom_out[3 * evt + 2] = p * uz;
    }
  }
}

extern "C" void launch_gKernel_Vertexing(
    const float* dTracks,
    const unsigned int* dNTracks,
    float* dPlane,
    int nEvents,
    float* dVtxOut,
    float* dMomOut,
    float* dChi2Out)
{
  dim3 block(128);
  dim3 grid((nEvents + block.x - 1) / block.x);
  gKernel_Vertexing<<<grid, block>>>(dTracks, dNTracks, dPlane, nEvents, dVtxOut, dMomOut, dChi2Out);
  cudaDeviceSynchronize();
}

