# utils/gpu_kernels.py - ER + XZ + YZ + Global wrappers (shared libs)
import numpy as np
import cupy as cp
import os, ctypes

MAX_HITS_ER = 1000
MAX_HITS_PER_EVENT = 512
MAX_TRACKLETS_PER_EVENT = 128
TRACKLET_FEATURES = 160

# Event Reducer
_evt_path = os.path.join(os.path.dirname(__file__), "libevent_reducer.so")
try:
    _event_reducer = np.ctypeslib.load_library(_evt_path, ".")
    _has_er = True
except OSError:
    _event_reducer = None
    _has_er = False

if _has_er:
    _event_reducer.launch_event_reducer.argtypes = [
        np.ctypeslib.ndpointer(np.float32, ndim=3, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.bool_, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ]

def run_event_reduction(hit_data, hit_flags, hastoomanyhits, datasizes):
    if not _has_er:
        return
    n_events = hit_data.shape[0]
    datasizes_kernel = np.asarray(datasizes, dtype=np.int32).copy()
    datasizes_kernel[0] = n_events
    _event_reducer.launch_event_reducer(hit_data, hit_flags, hastoomanyhits, datasizes_kernel)

# XZ
_lib_xz = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libxz_tracking.so"))
_lib_xz.launch_gKernel_XZ_tracking.restype = None
_lib_xz.launch_gKernel_XZ_tracking.argtypes = [
    ctypes.c_void_p,  # dHits
    ctypes.c_void_p,  # dNHits
    ctypes.c_void_p,  # dTracklets
    ctypes.c_void_p,  # dPlane
    ctypes.c_void_p,  # dNTracklets
    ctypes.c_void_p,  # dTooMany
    ctypes.c_int,     # nEvents
]

def run_xz_tracking(hit_vars, gplane_array, hastoomany_trim):
    assert gplane_array.dtype == np.float32 and gplane_array.ndim == 1
    n_events = len(hit_vars["detectorID"])
    counts = np.zeros(n_events, dtype=np.int32)
    gHits_np = np.zeros((n_events, 6 * MAX_HITS_PER_EVENT), dtype=np.float32)
    XZ_CHAMBERS = np.array([15, 16, 21, 22, 27, 28], dtype=np.int32)
    for evt in range(n_events):
        det_all = np.asarray(hit_vars["detectorID"][evt], dtype=np.int32)
        elm_all = np.asarray(hit_vars["elementID"][evt], dtype=np.float32)
        tdc_all = np.asarray(hit_vars["tdcTime"][evt], dtype=np.float32)
        drf_all = np.asarray(hit_vars["driftDistance"][evt], dtype=np.float32)
        mask = np.isin(det_all, XZ_CHAMBERS)
        det, elm, tdc, drf = det_all[mask], elm_all[mask], tdc_all[mask], drf_all[mask]
        nh = int(det.size)
        counts[evt] = min(nh, MAX_HITS_PER_EVENT)
        if nh:
            nh = counts[evt]
            base = gHits_np[evt]
            base[0 * nh:1 * nh] = det[:nh].astype(np.float32)
            base[1 * nh:2 * nh] = elm[:nh]
            base[2 * nh:3 * nh] = tdc[:nh]
            base[3 * nh:4 * nh] = 1.0
            base[4 * nh:5 * nh] = drf[:nh]
    d_gHits = cp.asarray(gHits_np, dtype=cp.float32)
    d_nHits = cp.asarray(counts, dtype=cp.int32)
    d_gPlane = cp.asarray(gplane_array, dtype=cp.float32)
    d_gTracks = cp.zeros((n_events, MAX_TRACKLETS_PER_EVENT, TRACKLET_FEATURES), dtype=cp.float32)
    d_nTracklets = cp.zeros(n_events, dtype=cp.uint32)
    d_tooMany = cp.asarray(hastoomany_trim, dtype=cp.bool_)
    _lib_xz.launch_gKernel_XZ_tracking(
        int(d_gHits.data.ptr),
        int(d_nHits.data.ptr),
        int(d_gTracks.data.ptr),
        int(d_gPlane.data.ptr),
        int(d_nTracklets.data.ptr),
        int(d_tooMany.data.ptr),
        int(n_events),
    )
    cp.cuda.runtime.deviceSynchronize()
    return (cp.asnumpy(d_gTracks), cp.asnumpy(d_nTracklets))

# YZ
_lib_yz = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libyz_tracking.so"))
_lib_yz.launch_gKernel_YZ_tracking.restype = None
_lib_yz.launch_gKernel_YZ_tracking.argtypes = [
    ctypes.c_void_p,  # dHits
    ctypes.c_void_p,  # dNHits
    ctypes.c_void_p,  # dPlane
    ctypes.c_void_p,  # dXZ seeds
    ctypes.c_void_p,  # dNXZ
    ctypes.c_void_p,  # dYZ out
    ctypes.c_void_p,  # dNYZ out
    ctypes.c_void_p,  # dTooMany
    ctypes.c_int,     # nEvents
]

def run_yz_tracking(hit_vars, gplane_array, hastoomany_trim, tracks_xz, ntracks_xz):
    assert gplane_array.dtype == np.float32 and gplane_array.ndim == 1
    n_events = len(hit_vars["detectorID"])
    counts = np.zeros(n_events, dtype=np.int32)
    gHits_np = np.zeros((n_events, 6 * MAX_HITS_PER_EVENT), dtype=np.float32)
    YZ_CHAMBERS = np.array([13, 14, 17, 18, 19, 20, 23, 24, 25, 26, 29, 30], dtype=np.int32)
    for evt in range(n_events):
        det_all = np.asarray(hit_vars["detectorID"][evt], dtype=np.int32)
        elm_all = np.asarray(hit_vars["elementID"][evt], dtype=np.float32)
        tdc_all = np.asarray(hit_vars["tdcTime"][evt], dtype=np.float32)
        drf_all = np.asarray(hit_vars["driftDistance"][evt], dtype=np.float32)
        mask = np.isin(det_all, YZ_CHAMBERS)
        det, elm, tdc, drf = det_all[mask], elm_all[mask], tdc_all[mask], drf_all[mask]
        nh = int(det.size)
        counts[evt] = min(nh, MAX_HITS_PER_EVENT)
        if nh:
            nh = counts[evt]
            base = gHits_np[evt]
            base[0 * nh:1 * nh] = det[:nh].astype(np.float32)
            base[1 * nh:2 * nh] = elm[:nh]
            base[2 * nh:3 * nh] = tdc[:nh]
            base[3 * nh:4 * nh] = 1.0
            base[4 * nh:5 * nh] = drf[:nh]
    d_gHits = cp.asarray(gHits_np, dtype=cp.float32)
    d_nHits = cp.asarray(counts, dtype=cp.int32)
    d_gPlane = cp.asarray(gplane_array, dtype=cp.float32)
    d_xzSeeds = cp.asarray(tracks_xz, dtype=cp.float32)
    d_nxz = cp.asarray(ntracks_xz, dtype=cp.uint32)
    d_yzOut = cp.zeros_like(d_xzSeeds)
    d_nyz = cp.zeros(n_events, dtype=cp.uint32)
    d_tooMany = cp.asarray(hastoomany_trim, dtype=cp.bool_)
    _lib_yz.launch_gKernel_YZ_tracking(
        int(d_gHits.data.ptr),
        int(d_nHits.data.ptr),
        int(d_gPlane.data.ptr),
        int(d_xzSeeds.data.ptr),
        int(d_nxz.data.ptr),
        int(d_yzOut.data.ptr),
        int(d_nyz.data.ptr),
        int(d_tooMany.data.ptr),
        int(n_events),
    )
    cp.cuda.runtime.deviceSynchronize()
    return (cp.asnumpy(d_yzOut), cp.asnumpy(d_nyz))

# Global
_lib_glb = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libglobal_tracking.so"))
_lib_glb.launch_gKernel_Global_Combine.restype = None
_lib_glb.launch_gKernel_Global_Combine.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,  # dXZ, dNXZ
    ctypes.c_void_p, ctypes.c_void_p,  # dYZ, dNYZ
    ctypes.c_void_p, ctypes.c_void_p,  # dOut, dNOut
    ctypes.c_int,                      # nEvents
    ctypes.c_float,                    # z_ref
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # dHits, dNHits, dPlane
]

def _pack_all_hits(hit_vars):
    n_events = len(hit_vars["detectorID"])
    counts = np.zeros(n_events, dtype=np.int32)
    gHits_np = np.zeros((n_events, 6 * MAX_HITS_PER_EVENT), dtype=np.float32)
    for evt in range(n_events):
        det = np.asarray(hit_vars["detectorID"][evt], dtype=np.int32)
        elm = np.asarray(hit_vars["elementID"][evt], dtype=np.float32)
        tdc = np.asarray(hit_vars["tdcTime"][evt], dtype=np.float32)
        drf = np.asarray(hit_vars["driftDistance"][evt], dtype=np.float32)
        nh = int(det.size)
        counts[evt] = min(nh, MAX_HITS_PER_EVENT)
        if nh:
            nh = counts[evt]
            base = gHits_np[evt]
            base[0 * nh:1 * nh] = det[:nh].astype(np.float32)
            base[1 * nh:2 * nh] = elm[:nh]
            base[2 * nh:3 * nh] = tdc[:nh]
            base[3 * nh:4 * nh] = 1.0
            base[4 * nh:5 * nh] = drf[:nh]
    return gHits_np, counts

def run_global_combine(tracks_xz, ntracks_xz, tracks_yz, ntracks_yz, hit_vars, gplane_array, z_ref=0.0, d0_ids=(3, 4), d0_win_cm=1.5, pt_kick=0.0075):
    n_events = tracks_xz.shape[0]
    d_xz = cp.asarray(tracks_xz, dtype=cp.float32)
    d_nx = cp.asarray(ntracks_xz, dtype=cp.uint32)
    d_yz = cp.asarray(tracks_yz, dtype=cp.float32)
    d_ny = cp.asarray(ntracks_yz, dtype=cp.uint32)
    d_out = cp.zeros_like(d_xz)
    d_no = cp.zeros(n_events, dtype=cp.uint32)
    gHits_np, counts = _pack_all_hits(hit_vars)
    d_hits = cp.asarray(gHits_np, dtype=cp.float32)
    d_nhits = cp.asarray(counts, dtype=cp.int32)
    d_plane = cp.asarray(gplane_array, dtype=cp.float32)
    _lib_glb.launch_gKernel_Global_Combine(
        int(d_xz.data.ptr), int(d_nx.data.ptr),
        int(d_yz.data.ptr), int(d_ny.data.ptr),
        int(d_out.data.ptr), int(d_no.data.ptr),
        int(n_events), float(z_ref),
        int(d_hits.data.ptr), int(d_nhits.data.ptr), int(d_plane.data.ptr),
    )
    cp.cuda.runtime.deviceSynchronize()
    return (cp.asnumpy(d_out), cp.asnumpy(d_no))
