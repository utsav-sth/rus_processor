#ifndef GPLANE_CUH
#define GPLANE_CUH

// Keep this aligned with utils/geometry.py
#define nDetectors 64

struct gPlane {
    float z[nDetectors];
    int   nelem[nDetectors];
    float cellwidth[nDetectors];
    float spacing[nDetectors];
    float xoffset[nDetectors];
    float scalex[nDetectors];
    float x0[nDetectors];
    float x1[nDetectors];
    float x2[nDetectors];
    float costheta[nDetectors];
    float scaley[nDetectors];
    float y0[nDetectors];
    float y1[nDetectors];
    float y2[nDetectors];
    float sintheta[nDetectors];
    float resolution[nDetectors];
    float p1x_w1[nDetectors];
    float p1y_w1[nDetectors];
    float p1z_w1[nDetectors];
    float deltapx[nDetectors];
    float deltapy[nDetectors];
    float deltapz[nDetectors];
    float dp1x[nDetectors];
    float dp1y[nDetectors];
    float dp1z[nDetectors];
    float slope_max[nDetectors];
    float inter_max[nDetectors];
};

#endif // GPLANE_CUH

