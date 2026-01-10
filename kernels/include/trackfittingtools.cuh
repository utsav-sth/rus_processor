#include "operations.h"
#pragma once
__device__ void chi2_simplefit(size_t const n_points, float* const x_array, float* const y_array, float& a, float& b, float sum, float det, float sx, float sy, float sxx, float syy, float sxy) {
    sx = 0;
    sy = 0;
    sxx = 0;
    syy = 0;
    sxy = 0;
    for (int i = 0; i < n_points; i++) {
        sum += 1.0f;
        sx += x_array[i];
        sy += y_array[i];
        sxx += x_array[i] * x_array[i];
        syy += y_array[i] * y_array[i];
        sxy += x_array[i] * y_array[i];
    }
    det = sum * sxx - sx * sx;
    if (fabs(det) < 1.0e-20f) {
        a = 0.0f;
        b = 0.0f;
        return;
    }
    a = (sum * sxy - sx * sy) / det;
    b = (sy * sxx - sxy * sx) / det;
}

__device__ void fit_2D_track(size_t const n_points, const float* x_points, const float* z_points, const float* x_weights, float* A, float* Ainv, float* B, float* output_parameters, float* output_parameters_errors, float& chi2) {
    for (int j = 0; j < 2; j++) {
        B[j] = 0;
        for (int k = 0; k < 2; k++) {
            A[j * 2 + k] = 0;
            Ainv[j * 2 + k] = 0;
        }
    }
    for (int i = 0; i < n_points; i++) {
        B[0] += x_weights[i] * x_points[i];
        B[1] += x_weights[i] * x_points[i] * z_points[i];
        A[0] += x_weights[i];
        A[1] += x_weights[i] * z_points[i];
        A[2] += x_weights[i] * z_points[i];
        A[3] += x_weights[i] * z_points[i] * z_points[i];
    }
    matinv_2x2_matrix_per_thread(A, Ainv);
    for (int j = 0; j < 2; j++) {
        output_parameters[j] = 0.0f;
        output_parameters_errors[j] = sqrtf(fabs(Ainv[j * 2 + j]));
        for (int k = 0; k < 2; k++) {
            output_parameters[j] += Ainv[j * 2 + k] * B[k];
        }
    }
    chi2 = 0.0f;
    for (int i = 0; i < n_points; i++) {
        float pred = output_parameters[0] + z_points[i] * output_parameters[1];
        chi2 += x_weights[i] * x_weights[i] * pred * pred;
    }
}
