#pragma once
#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef absOp
__device__ __forceinline__ float absOp(float x) {
    return fabsf(x);
}
#endif

#ifndef rcpOp
__device__ __forceinline__ float rcpOp(float x) {
    return 1.0f / x;
}
#endif

#ifndef mulOp
__device__ __forceinline__ float mulOp(float a, float b) {
    return a * b;
}
#endif

#ifndef negOp
__device__ __forceinline__ float negOp(float x) {
    return -x;
}
#endif

#ifndef fmnaOp
__device__ __forceinline__ float fmnaOp(float a, float b, float c) {
    return c - a * b;
}
#endif

#ifdef __CUDACC__
static __device__ __forceinline__ void matinv_2x2_matrix_per_thread(const float*, float*);
static __device__ __forceinline__ void matinv_4x4_matrix_per_thread(const float*, float*);
#endif

static __device__ __forceinline__ void matinv_2x2_matrix_per_thread(const float* A, float* Ainv) {
    int perm0, perm1, icol0, icol1, i, pvt;
    float AA00, AA01, AA10, AA11, tmp, t, p;

    AA00 = A[0]; AA10 = A[1];
    AA01 = A[2]; AA11 = A[3];

    perm0 = 0; perm1 = 1;

    p = absOp(AA00); pvt = 0;
    t = absOp(AA10);
    if (t > p) {
        p = t; pvt = 1;
    }

    if (pvt == 1) {
        tmp = AA00; AA00 = AA10; AA10 = tmp;
        tmp = AA01; AA01 = AA11; AA11 = tmp;
        i = perm0; perm0 = perm1; perm1 = i;
    }

    tmp = rcpOp(AA00); icol0 = perm0;
    AA00 = tmp; AA01 = mulOp(tmp, AA01);

    tmp = AA10;
    AA10 = mulOp(negOp(tmp), AA00);
    AA11 = fmnaOp(tmp, AA01, AA11);

    tmp = rcpOp(AA11); icol1 = perm1;
    AA10 = mulOp(tmp, AA10); AA11 = tmp;

    tmp = AA01;
    AA00 = fmnaOp(tmp, AA10, AA00);
    AA01 = mulOp(negOp(tmp), AA11);

    Ainv[0 * 2 + icol0] = AA00; Ainv[1 * 2 + icol0] = AA10;
    Ainv[0 * 2 + icol1] = AA01; Ainv[1 * 2 + icol1] = AA11;
}

static __device__ __forceinline__ void matinv_4x4_matrix_per_thread(const float* A, float* Ainv) {
    int perm0, perm1, perm2, perm3;
    int icol0, icol1, icol2, icol3, i, pvt;
    float AA00, AA01, AA02, AA03,
          AA10, AA11, AA12, AA13,
          AA20, AA21, AA22, AA23,
          AA30, AA31, AA32, AA33,
          tmp, t, p;

    AA00 = A[0];  AA10 = A[1];  AA20 = A[2];  AA30 = A[3];
    AA01 = A[4];  AA11 = A[5];  AA21 = A[6];  AA31 = A[7];
    AA02 = A[8];  AA12 = A[9];  AA22 = A[10]; AA32 = A[11];
    AA03 = A[12]; AA13 = A[13]; AA23 = A[14]; AA33 = A[15];

    perm0 = 0; perm1 = 1; perm2 = 2; perm3 = 3;

    p = absOp(AA00); pvt = 0;
    t = absOp(AA10);
    if (t > p) {
        p = t; pvt = 1;
    }
    t = absOp(AA20);
    if (t > p) {
        p = t; pvt = 2;
    }
    t = absOp(AA30);
    if (t > p) {
        p = t; pvt = 3;
    }

    if (pvt == 1) {
        tmp = AA00; AA00 = AA10; AA10 = tmp;
        tmp = AA01; AA01 = AA11; AA11 = tmp;
        tmp = AA02; AA02 = AA12; AA12 = tmp;
        tmp = AA03; AA03 = AA13; AA13 = tmp;
        i = perm0; perm0 = perm1; perm1 = i;
    } else if (pvt == 2) {
        tmp = AA00; AA00 = AA20; AA20 = tmp;
        tmp = AA01; AA01 = AA21; AA21 = tmp;
        tmp = AA02; AA02 = AA22; AA22 = tmp;
        tmp = AA03; AA03 = AA23; AA23 = tmp;
        i = perm0; perm0 = perm2; perm2 = i;
    } else if (pvt == 3) {
        tmp = AA00; AA00 = AA30; AA30 = tmp;
        tmp = AA01; AA01 = AA31; AA31 = tmp;
        tmp = AA02; AA02 = AA32; AA32 = tmp;
        tmp = AA03; AA03 = AA33; AA33 = tmp;
        i = perm0; perm0 = perm3; perm3 = i;
    }

    tmp = rcpOp(AA00); icol0 = perm0;
    AA00 = tmp;
    AA01 = mulOp(tmp, AA01);
    AA02 = mulOp(tmp, AA02);
    AA03 = mulOp(tmp, AA03);

    tmp = AA10;
    AA10 = mulOp(negOp(tmp), AA00);
    AA11 = fmnaOp(tmp, AA01, AA11);
    AA12 = fmnaOp(tmp, AA02, AA12);
    AA13 = fmnaOp(tmp, AA03, AA13);

    tmp = AA20;
    AA20 = mulOp(negOp(tmp), AA00);
    AA21 = fmnaOp(tmp, AA01, AA21);
    AA22 = fmnaOp(tmp, AA02, AA22);
    AA23 = fmnaOp(tmp, AA03, AA23);

    tmp = AA30;
    AA30 = mulOp(negOp(tmp), AA00);
    AA31 = fmnaOp(tmp, AA01, AA31);
    AA32 = fmnaOp(tmp, AA02, AA32);
    AA33 = fmnaOp(tmp, AA03, AA33);

    p = absOp(AA11); pvt = 1;
    t = absOp(AA21);
    if (t > p) {
        p = t; pvt = 2;
    }
    t = absOp(AA31);
    if (t > p) {
        p = t; pvt = 3;
    }

    if (pvt == 2) {
        tmp = AA10; AA10 = AA20; AA20 = tmp;
        tmp = AA11; AA11 = AA21; AA21 = tmp;
        tmp = AA12; AA12 = AA22; AA22 = tmp;
        tmp = AA13; AA13 = AA23; AA23 = tmp;
        i = perm1; perm1 = perm2; perm2 = i;
    } else if (pvt == 3) {
        tmp = AA10; AA10 = AA30; AA30 = tmp;
        tmp = AA11; AA11 = AA31; AA31 = tmp;
        tmp = AA12; AA12 = AA32; AA32 = tmp;
        tmp = AA13; AA13 = AA33; AA33 = tmp;
        i = perm1; perm1 = perm3; perm3 = i;
    }

    tmp = rcpOp(AA11); icol1 = perm1;
    AA10 = mulOp(tmp, AA10); AA11 = tmp;
    AA12 = mulOp(tmp, AA12); AA13 = mulOp(tmp, AA13);

    tmp = AA01;
    AA00 = fmnaOp(tmp, AA10, AA00);
    AA01 = mulOp(negOp(tmp), AA11);
    AA02 = fmnaOp(tmp, AA12, AA02);
    AA03 = fmnaOp(tmp, AA13, AA03);

    tmp = AA21;
    AA20 = fmnaOp(tmp, AA10, AA20);
    AA21 = mulOp(negOp(tmp), AA11);
    AA22 = fmnaOp(tmp, AA12, AA22);
    AA23 = fmnaOp(tmp, AA13, AA23);

    tmp = AA31;
    AA30 = fmnaOp(tmp, AA10, AA30);
    AA31 = mulOp(negOp(tmp), AA11);
    AA32 = fmnaOp(tmp, AA12, AA32);
    AA33 = fmnaOp(tmp, AA13, AA33);

    p = absOp(AA22); pvt = 2;
    t = absOp(AA32);
    if (t > p) {
        p = t; pvt = 3;
    }

    if (pvt == 3) {
        tmp = AA20; AA20 = AA30; AA30 = tmp;
        tmp = AA21; AA21 = AA31; AA31 = tmp;
        tmp = AA22; AA22 = AA32; AA32 = tmp;
        tmp = AA23; AA23 = AA33; AA33 = tmp;
        i = perm2; perm2 = perm3; perm3 = i;
    }

    tmp = rcpOp(AA22); icol2 = perm2;
    AA20 = mulOp(tmp, AA20); AA21 = mulOp(tmp, AA21);
    AA22 = tmp; AA23 = mulOp(tmp, AA23);

    tmp = AA02;
    AA00 = fmnaOp(tmp, AA20, AA00);
    AA01 = fmnaOp(tmp, AA21, AA01);
    AA02 = mulOp(negOp(tmp), AA22);
    AA03 = fmnaOp(tmp, AA23, AA03);

    tmp = AA12;
    AA10 = fmnaOp(tmp, AA20, AA10);
    AA11 = fmnaOp(tmp, AA21, AA11);
    AA12 = mulOp(negOp(tmp), AA22);
    AA13 = fmnaOp(tmp, AA23, AA13);

    tmp = AA32;
    AA30 = fmnaOp(tmp, AA20, AA30);
    AA31 = fmnaOp(tmp, AA21, AA31);
    AA32 = mulOp(negOp(tmp), AA22);
    AA33 = fmnaOp(tmp, AA23, AA33);

    tmp = rcpOp(AA33); icol3 = perm3;
    AA30 = mulOp(tmp, AA30); AA31 = mulOp(tmp, AA31);
    AA32 = mulOp(tmp, AA32); AA33 = tmp;

    tmp = AA03;
    AA00 = fmnaOp(tmp, AA30, AA00);
    AA01 = fmnaOp(tmp, AA31, AA01);
    AA02 = fmnaOp(tmp, AA32, AA02);
    AA03 = mulOp(negOp(tmp), AA33);

    tmp = AA13;
    AA10 = fmnaOp(tmp, AA30, AA10);
    AA11 = fmnaOp(tmp, AA31, AA11);
    AA12 = fmnaOp(tmp, AA32, AA12);
    AA13 = mulOp(negOp(tmp), AA33);

    tmp = AA23;
    AA20 = fmnaOp(tmp, AA30, AA20);
    AA21 = fmnaOp(tmp, AA31, AA21);
    AA22 = fmnaOp(tmp, AA32, AA22);
    AA23 = mulOp(negOp(tmp), AA33);

    Ainv[0 * 4 + icol0] = AA00; Ainv[1 * 4 + icol0] = AA10;
    Ainv[2 * 4 + icol0] = AA20; Ainv[3 * 4 + icol0] = AA30;

    Ainv[0 * 4 + icol1] = AA01; Ainv[1 * 4 + icol1] = AA11;
    Ainv[2 * 4 + icol1] = AA21; Ainv[3 * 4 + icol1] = AA31;

    Ainv[0 * 4 + icol2] = AA02; Ainv[1 * 4 + icol2] = AA12;
    Ainv[2 * 4 + icol2] = AA22; Ainv[3 * 4 + icol2] = AA32;

    Ainv[0 * 4 + icol3] = AA03; Ainv[1 * 4 + icol3] = AA13;
    Ainv[2 * 4 + icol3] = AA23; Ainv[3 * 4 + icol3] = AA33;
}

__device__ float chi2_track(size_t const n_points,
                            float* const driftdist, short* const sign,
                            float* const resolutions,
                            float* const p1x, float* const p1y, float* const p1z,
                            float* const deltapx, float* const deltapy, float* const deltapz,
                            const float x0, const float y0,
                            const float tx, const float ty) {
    float chi2 = 0.f, dca, den2;
    for (size_t i = 0; i < n_points; ++i) {
        den2 = deltapy[i] * deltapy[i] * (1 + tx * tx)
             + deltapx[i] * deltapx[i] * (1 + ty * ty)
             - 2 * (ty * deltapx[i] * deltapz[i] + ty * deltapy[i] * deltapz[i] + tx * ty * deltapx[i] * deltapy[i]);

        dca = ((ty * deltapz[i] - deltapy[i]) * (p1x[i] - x0)
             + (deltapx[i] - tx * deltapz[i]) * (p1y[i] - y0)
             + p1z[i] * (tx * deltapy[i] - ty * deltapx[i])) / sqrtf(den2);

        float diff = driftdist[i] * sign[i] - dca;
        chi2 += diff * diff / (resolutions[i] * resolutions[i]);
    }
    return chi2;
}

#endif
