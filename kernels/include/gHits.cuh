// File: kernels/include/gHits.cuh
#ifndef GHITS_CUH
#define GHITS_CUH

#include <assert.h>
#include <stddef.h>

struct gHits {
    const unsigned int NHitsTotal;
    float* m_hitdata;

    __host__ __device__
    gHits(float* basedata, const unsigned total_number_of_hits, const unsigned offset = 0)
        : m_hitdata(reinterpret_cast<float*>(basedata) + offset), NHitsTotal(total_number_of_hits) {
        static_assert(sizeof(float) == sizeof(unsigned), "float size mismatch");
        assert(basedata != nullptr);
    }

    __host__ __device__ inline float chan(const unsigned index) const {
        assert(index < NHitsTotal);
        return m_hitdata[index];
    }

    __host__ __device__ inline float pos(const unsigned index) const {
        assert(index < NHitsTotal);
        return m_hitdata[NHitsTotal + index];
    }

    __host__ __device__ inline float tdc(const unsigned index) const {
        assert(index < NHitsTotal);
        return m_hitdata[NHitsTotal * 2 + index];
    }

    __host__ __device__ inline float flag(const unsigned index) const {
        assert(index < NHitsTotal);
        return m_hitdata[NHitsTotal * 3 + index];
    }

    __host__ __device__ inline float drift(const unsigned index) const {
        assert(index < NHitsTotal);
        return m_hitdata[NHitsTotal * 4 + index];
    }
};

#endif // GHITS_CUH

