#ifndef GTRACKLET_CUH
#define GTRACKLET_CUH

#include <assert.h>

struct gTracklet {
    float* __restrict__ m_trackletdata;

    __host__ __device__ gTracklet(float* basedata, unsigned offset = 0)
        : m_trackletdata(basedata + offset) {
        static_assert(sizeof(float) == 4, "Unexpected float size");
    }

    __host__ __device__ inline float stationID() const {
        return m_trackletdata[0];
    }
    __host__ __device__ inline float threadID() const {
        return m_trackletdata[1];
    }
    __host__ __device__ inline float nHits() const {
        return m_trackletdata[2];
    }
    __host__ __device__ inline float chisq() const {
        return m_trackletdata[3];
    }
    __host__ __device__ inline float chisq_vtx() const {
        return m_trackletdata[4];
    }

    __host__ __device__ inline float tx() const {
        return m_trackletdata[5];
    }
    __host__ __device__ inline float ty() const {
        return m_trackletdata[6];
    }
    __host__ __device__ inline float x0() const {
        return m_trackletdata[7];
    }
    __host__ __device__ inline float y0() const {
        return m_trackletdata[8];
    }
    __host__ __device__ inline float invP() const {
        return m_trackletdata[9];
    }

    __host__ __device__ inline float hits_detid(int i) const {
        return m_trackletdata[16 + i];
    }
    __host__ __device__ inline float hits_chan(int i) const {
        return m_trackletdata[34 + i];
    }
    __host__ __device__ inline float hits_pos(int i) const {
        return m_trackletdata[52 + i];
    }
    __host__ __device__ inline float hits_drift(int i) const {
        return m_trackletdata[70 + i];
    }
    __host__ __device__ inline float hits_sign(int i) const {
        return m_trackletdata[88 + i];
    }
    __host__ __device__ inline float hits_tdc(int i) const {
        return m_trackletdata[124 + i];
    }
    __host__ __device__ inline float hits_residual(int i) const {
        return m_trackletdata[142 + i];
    }

    __host__ __device__ inline void setStationID(float v) {
        m_trackletdata[0] = v;
    }
    __host__ __device__ inline void setThreadID(float v) {
        m_trackletdata[1] = v;
    }
    __host__ __device__ inline void setNHits(float v) {
        m_trackletdata[2] = v;
    }
    __host__ __device__ inline void setChisq(float v) {
        m_trackletdata[3] = v;
    }
    __host__ __device__ inline void setChisqVtx(float v) {
        m_trackletdata[4] = v;
    }

    __host__ __device__ inline void setTx(float v) {
        m_trackletdata[5] = v;
    }
    __host__ __device__ inline void setTy(float v) {
        m_trackletdata[6] = v;
    }
    __host__ __device__ inline void setX0(float v) {
        m_trackletdata[7] = v;
    }
    __host__ __device__ inline void setY0(float v) {
        m_trackletdata[8] = v;
    }
    __host__ __device__ inline void setInvP(float v) {
        m_trackletdata[9] = v;
    }

    __host__ __device__ inline void setHitDetID(int i, float v) {
        m_trackletdata[16 + i] = v;
    }
    __host__ __device__ inline void setHitChan(int i, float v) {
        m_trackletdata[34 + i] = v;
    }
    __host__ __device__ inline void setHitPos(int i, float v) {
        m_trackletdata[52 + i] = v;
    }
    __host__ __device__ inline void setHitDrift(int i, float v) {
        m_trackletdata[70 + i] = v;
    }
    __host__ __device__ inline void setHitSign(int i, float v) {
        m_trackletdata[88 + i] = v;
    }
    __host__ __device__ inline void setHitTDC(int i, float v) {
        m_trackletdata[124 + i] = v;
    }
    __host__ __device__ inline void setHitResidual(int i, float v) {
        m_trackletdata[142 + i] = v;
    }

    __host__ __device__ inline float* raw() {
        return m_trackletdata;
    }
};

#endif
