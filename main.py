# (full file content exactly as updated)
# main.py — ER -> XZ -> YZ (per XZ) -> Global combine -> Vertexing -> Dimuon -> write
import yaml
import awkward as ak
import numpy as np

from utils.reader import read_rus_file
from utils.writer import write_rus_file
from utils.geometry import load_gplane, load_gplane_with_vertex_constants
from utils.gpu_kernels import (
    run_event_reduction,
    run_xz_tracking,
    run_yz_tracking,
    run_global_combine,
    run_vertexing,
    run_dimuon_building,
    MAX_HITS_ER,
)

# hit feature indices (used only for ER packing here)
FEATURE_ELEMENT_ID = 0
FEATURE_DETECTOR_ID = 1
FEATURE_WIRE_POS = 2
FEATURE_TDC = 3
FEATURE_DRIFT = 4
FEATURE_INTIME = 5

# ------------------------------ config ------------------------------
with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

input_file = config["input_file"]
output_file = config["output_file"]

er_enabled = config.get("event_reducer", {}).get("enabled", True)
xz_enabled = config.get("xz_tracklet_builder", {}).get("enabled", True)
yz_enabled = config.get("yz_tracklet_builder", {}).get("enabled", True)

glb_cfg = config.get("global_tracking", {})
vtx_cfg = config.get("vertexing", {})
vtx_enabled = bool(vtx_cfg.get("enabled", True))

dimu_cfg = config.get("dimuon", {})
dimu_enabled = bool(dimu_cfg.get("enabled", False))
dimu_mass_min = float(dimu_cfg.get("mass_min", 0.0))
dimu_mass_max = float(dimu_cfg.get("mass_max", 999.0))
dimu_require_opp = bool(dimu_cfg.get("require_opposite_charge", True))

# global KMAG anchor
z_kmag_bend = glb_cfg.get("z_kmag_bend_cm", None)
if z_kmag_bend is None:
    z_kmag_bend = float("nan")
    print("[global] KMAG anchor: DISABLED (z_kmag_bend_cm is null/missing)")
else:
    z_kmag_bend = float(z_kmag_bend)
    print(f"[global] KMAG anchor: ENABLED at z = {z_kmag_bend:.3f} cm")

d0_x_ids = glb_cfg.get("d0_x_det_ids", [3, 4])
d0_x_window = float(glb_cfg.get("d0_x_window_cm", 1.5))        # cm
pt_kick_kmag = float(glb_cfg.get("pt_kick_kmag", 0.3819216))   # GeV/c (KTracker-like)
require_both = bool(glb_cfg.get("require_both_views", True))

# ------------------------------ input ------------------------------
event_vars, hit_vars, track_vars_in = read_rus_file(input_file)

required_hits = ["detectorID", "elementID", "tdcTime", "driftDistance"]
if not all(k in hit_vars for k in required_hits):
    raise RuntimeError("Missing required hit variables for event reduction / tracking.")

N_EVENTS = len(event_vars["eventID"])
MAX_HITS_FILE = max(len(ev) for ev in hit_vars["detectorID"])
MAX_HITS_BUF = max(MAX_HITS_FILE, MAX_HITS_ER)

# fabricate hitsInTime if missing
if "hitsInTime" not in hit_vars:
    print("Warning: 'hitsInTime' not found in tree - skipped.")
    print("Info: fabricated default hitsInTime=True for every hit.")
    hit_vars["hitsInTime"] = ak.Array([[True] * len(ev) for ev in hit_vars["detectorID"]])

# pack ER input (AoS-ish for ER kernel)
datasizes = np.array([len(ev) for ev in hit_vars["detectorID"]], dtype=np.int32)
hit_data = np.zeros((N_EVENTS, MAX_HITS_BUF, 6), dtype=np.float32)
hit_flags = np.ones((N_EVENTS, MAX_HITS_BUF), dtype=np.int32)
hastoomany = np.zeros(N_EVENTS, dtype=bool)

for i_evt in range(N_EVENTS):
    n_hits_evt = len(hit_vars["detectorID"][i_evt])
    for j in range(n_hits_evt):
        detid = int(hit_vars["detectorID"][i_evt][j])
        elemid = int(hit_vars["elementID"][i_evt][j])
        hit_data[i_evt, j, FEATURE_ELEMENT_ID] = elemid
        hit_data[i_evt, j, FEATURE_DETECTOR_ID] = detid
        hit_data[i_evt, j, FEATURE_WIRE_POS] = 0.0
        hit_data[i_evt, j, FEATURE_TDC] = hit_vars["tdcTime"][i_evt][j]
        hit_data[i_evt, j, FEATURE_DRIFT] = hit_vars["driftDistance"][i_evt][j]
        hit_data[i_evt, j, FEATURE_INTIME] = 1.0

# ------------------------------ ER ------------------------------
if er_enabled:
    run_event_reduction(hit_data, hit_flags, hastoomany, datasizes)
    print("Event reduction complete.\n")
else:
    print("Event-reducer disabled - skipping GPU step.")

# prune
kept_mask = ~hastoomany

if er_enabled:
    for k in list(event_vars.keys()):
        arr = np.asarray(event_vars[k])
        if len(arr) == len(kept_mask):
            event_vars[k] = arr[kept_mask]

pruned = {k: [] for k in hit_vars.keys()}
for i_evt in range(N_EVENTS):
    if not kept_mask[i_evt]:
        continue
    n_hits_evt = len(hit_vars["detectorID"][i_evt])
    mask = hit_flags[i_evt, :n_hits_evt] > 0
    for key in hit_vars:
        arr = np.asarray(hit_vars[key][i_evt])
        pruned[key].append(arr[mask].tolist())

hit_vars = {k: ak.Array(v) for k, v in pruned.items()}
n_evt_after = len(hit_vars["detectorID"])
print(f"Kept {n_evt_after} / {N_EVENTS} events after reduction.")

# ------------------------------ geometry blob ------------------------------
need_vertex_constants = (vtx_enabled or dimu_enabled)
if need_vertex_constants:
    gplane_array = load_gplane_with_vertex_constants("config/plane.txt", config)
else:
    gplane_array = load_gplane("config/plane.txt")

# ------------------------------ tracking ------------------------------
tracks_xz = ntracks_xz = None
tracks_yz = ntracks_yz = None

if xz_enabled:
    hastoomany_trim = np.zeros(n_evt_after, dtype=bool)
    tracks_xz, ntracks_xz = run_xz_tracking(hit_vars, gplane_array, hastoomany_trim)

if yz_enabled:
    hastoomany_trim = np.zeros(n_evt_after, dtype=bool)
    if tracks_xz is None:
        tracks_yz = np.zeros((n_evt_after, 1, 160), np.float32)
        ntracks_yz = np.zeros(n_evt_after, np.uint32)
    else:
        tracks_yz, ntracks_yz = run_yz_tracking(hit_vars, gplane_array, hastoomany_trim, tracks_xz, ntracks_xz)

# ------------------------------ global combine ------------------------------
D0_X_IDS = set(int(d) for d in d0_x_ids)
d0_with_hits = sum(
    any(int(d) in D0_X_IDS for d in hit_vars["detectorID"][i])
    for i in range(len(hit_vars["detectorID"]))
)
print(f"[global] Events with any D0-X hits {sorted(D0_X_IDS)}: {d0_with_hits}/{n_evt_after}")

tracks_glb, ntracks_glb = run_global_combine(
    tracks_xz if tracks_xz is not None else np.zeros((n_evt_after, 1, 160), np.float32),
    ntracks_xz if ntracks_xz is not None else np.zeros((n_evt_after,), np.uint32),
    tracks_yz if tracks_yz is not None else np.zeros((n_evt_after, 1, 160), np.float32),
    ntracks_yz if ntracks_yz is not None else np.zeros((n_evt_after,), np.uint32),
    hit_vars, gplane_array,
    z_ref=z_kmag_bend,
    d0_ids=list(D0_X_IDS), d0_win_cm=d0_x_window, pt_kick=pt_kick_kmag,
)

# debug summaries
if ntracks_xz is not None:
    print(f"[XZ] events with >=2 tracklets: {int((ntracks_xz >= 2).sum())}/{n_evt_after}, max={int(ntracks_xz.max())}")
if ntracks_yz is not None:
    print(f"[YZ] events with >=2 tracklets: {int((ntracks_yz >= 2).sum())}/{n_evt_after}, max={int(ntracks_yz.max())}")

# ------------------------------ vertexing ------------------------------
vtx_out = mom_out = chi2_vtx = None
if vtx_enabled:
    vtx_out, mom_out, chi2_vtx = run_vertexing(tracks_glb, ntracks_glb, gplane_array)
    print("[vtx] vertexing kernel complete.")

# ------------------------------ dimuon building ------------------------------
dimu_ndimu = None
(dimu_vtx, dimu_mom, dimu_mass, dimu_chi2, mu1_pq, mu2_pq,
 dimu_xf, dimu_x1, dimu_x2, dimu_pt, dimu_phi, dimu_costh) = (None,) * 12

if dimu_enabled:
    if (tracks_xz is None) or (tracks_yz is None):
        print("[dimu] disabled: XZ/YZ tracklets are missing.")
        dimu_enabled = False
    elif not np.isfinite(z_kmag_bend):
        print("[dimu] disabled: z_kmag_bend_cm is not set/finite (required).")
        dimu_enabled = False
    else:
        try:
            (
                dimu_ndimu,
                dimu_vtx,
                dimu_mom,
                dimu_mass,
                dimu_chi2,
                mu1_pq,
                mu2_pq,
                dimu_xf,
                dimu_x1,
                dimu_x2,
                dimu_pt,
                dimu_phi,
                dimu_costh,
            ) = run_dimuon_building(
                tracks_xz, ntracks_xz,
                tracks_yz, ntracks_yz,
                hit_vars, gplane_array,
                z_kmag_bend_cm=z_kmag_bend,
                pt_kick_kmag=pt_kick_kmag,
                d0_win_cm=d0_x_window,
                mass_min=dimu_mass_min,
                mass_max=dimu_mass_max,
                require_opposite_charge=dimu_require_opp,
            )
            print("[dimu] dimuon kernel complete.")
            npos = int((dimu_ndimu > 0).sum())
            print(f"[dimu] kernel reports nDimu>0 in {npos}/{n_evt_after} events")
        except Exception as e:
            print(f"[dimu] ERROR: {type(e).__name__}: {e}")
            print("[dimu] disabling dimuon for this run so output still gets written.")
            dimu_enabled = False

# ------------------------------ pack combined tree ------------------------------
nan = float("nan")
track_vars_out = {
    "stationID": [],
    "x0": [], "tx": [], "nhits_xz": [], "chi2_xz": [],
    "y0": [], "ty": [], "nhits_yz": [], "chi2_yz": [],
    "invP": [],
    "vx": [], "vy": [], "vz": [],
    "px": [], "py": [], "pz": [],
    "chi2_upstream": [], "chi2_target": [], "chi2_dump": [],

    # dimuon outputs
    "dimu_vx": [], "dimu_vy": [], "dimu_vz": [],
    "dimu_px": [], "dimu_py": [], "dimu_pz": [],
    "dimu_mass": [], "dimu_chi2": [],
    "dimu_xF": [], "dimu_x1": [], "dimu_x2": [],
    "dimu_pT": [], "dimu_phi": [], "dimu_costheta": [],

    "mu1_px": [], "mu1_py": [], "mu1_pz": [], "mu1_q": [],
    "mu2_px": [], "mu2_py": [], "mu2_pz": [], "mu2_q": [],
}

xz_valid = yz_valid = both_valid = 0
dimu_valid = 0

for ievt in range(n_evt_after):
    nout = int(ntracks_glb[ievt])

    # XZ from glb track 0
    if nout >= 1 and np.isfinite(tracks_glb[ievt, 0, 3]):
        nhx = int(tracks_glb[ievt, 0, 2])
        chi2x = float(tracks_glb[ievt, 0, 3])
        tx = float(tracks_glb[ievt, 0, 5])
        x0 = float(tracks_glb[ievt, 0, 7])
        track_vars_out["nhits_xz"].append([nhx])
        track_vars_out["chi2_xz"].append([chi2x])
        track_vars_out["tx"].append([tx])
        track_vars_out["x0"].append([x0])
        xz_valid += 1
    else:
        track_vars_out["nhits_xz"].append([])
        track_vars_out["chi2_xz"].append([])
        track_vars_out["tx"].append([])
        track_vars_out["x0"].append([])

    # YZ from glb track 1
    if nout >= 2 and np.isfinite(tracks_glb[ievt, 1, 3]):
        nhy = int(tracks_glb[ievt, 1, 2])
        chi2y = float(tracks_glb[ievt, 1, 3])
        ty = float(tracks_glb[ievt, 1, 5])
        y0 = float(tracks_glb[ievt, 1, 7])
        track_vars_out["nhits_yz"].append([nhy])
        track_vars_out["chi2_yz"].append([chi2y])
        track_vars_out["ty"].append([ty])
        track_vars_out["y0"].append([y0])
        yz_valid += 1
    else:
        track_vars_out["nhits_yz"].append([])
        track_vars_out["chi2_yz"].append([])
        track_vars_out["ty"].append([])
        track_vars_out["y0"].append([])

    invp0 = tracks_glb[ievt, 0, 14] if (nout >= 1) else nan
    invp1 = tracks_glb[ievt, 1, 14] if (nout >= 2) else nan
    invp = invp0 if np.isfinite(invp0) else (invp1 if np.isfinite(invp1) else nan)
    track_vars_out["invP"].append([float(invp)]) if np.isfinite(invp) else track_vars_out["invP"].append([])

    track_vars_out["stationID"].append([1] if (len(track_vars_out["tx"][-1]) or len(track_vars_out["ty"][-1])) else [])

    # Vertexing
    if vtx_enabled and vtx_out is not None and nout >= 1 and np.isfinite(vtx_out[ievt, 2]):
        track_vars_out["vx"].append([float(vtx_out[ievt, 0])])
        track_vars_out["vy"].append([float(vtx_out[ievt, 1])])
        track_vars_out["vz"].append([float(vtx_out[ievt, 2])])
        track_vars_out["px"].append([float(mom_out[ievt, 0])])
        track_vars_out["py"].append([float(mom_out[ievt, 1])])
        track_vars_out["pz"].append([float(mom_out[ievt, 2])])
        track_vars_out["chi2_upstream"].append([float(chi2_vtx[ievt, 0])])
        track_vars_out["chi2_target"].append([float(chi2_vtx[ievt, 1])])
        track_vars_out["chi2_dump"].append([float(chi2_vtx[ievt, 2])])
    else:
        track_vars_out["vx"].append([]); track_vars_out["vy"].append([]); track_vars_out["vz"].append([])
        track_vars_out["px"].append([]); track_vars_out["py"].append([]); track_vars_out["pz"].append([])
        track_vars_out["chi2_upstream"].append([]); track_vars_out["chi2_target"].append([]); track_vars_out["chi2_dump"].append([])

    # Dimuon
    if dimu_enabled and dimu_ndimu is not None and int(dimu_ndimu[ievt]) > 0:
        m = float(dimu_mass[ievt])
        ok = np.isfinite(m) and (m >= dimu_mass_min) and (m <= dimu_mass_max) and np.isfinite(dimu_vtx[ievt, 2])
        if ok:
            track_vars_out["dimu_vx"].append([float(dimu_vtx[ievt, 0])])
            track_vars_out["dimu_vy"].append([float(dimu_vtx[ievt, 1])])
            track_vars_out["dimu_vz"].append([float(dimu_vtx[ievt, 2])])
            track_vars_out["dimu_px"].append([float(dimu_mom[ievt, 0])])
            track_vars_out["dimu_py"].append([float(dimu_mom[ievt, 1])])
            track_vars_out["dimu_pz"].append([float(dimu_mom[ievt, 2])])
            track_vars_out["dimu_mass"].append([m])
            track_vars_out["dimu_chi2"].append([float(dimu_chi2[ievt])])

            track_vars_out["dimu_xF"].append([float(dimu_xf[ievt])]) if np.isfinite(dimu_xf[ievt]) else track_vars_out["dimu_xF"].append([])
            track_vars_out["dimu_x1"].append([float(dimu_x1[ievt])]) if np.isfinite(dimu_x1[ievt]) else track_vars_out["dimu_x1"].append([])
            track_vars_out["dimu_x2"].append([float(dimu_x2[ievt])]) if np.isfinite(dimu_x2[ievt]) else track_vars_out["dimu_x2"].append([])
            track_vars_out["dimu_pT"].append([float(dimu_pt[ievt])]) if np.isfinite(dimu_pt[ievt]) else track_vars_out["dimu_pT"].append([])
            track_vars_out["dimu_phi"].append([float(dimu_phi[ievt])]) if np.isfinite(dimu_phi[ievt]) else track_vars_out["dimu_phi"].append([])
            track_vars_out["dimu_costheta"].append([float(dimu_costh[ievt])]) if np.isfinite(dimu_costh[ievt]) else track_vars_out["dimu_costheta"].append([])

            q1 = mu1_pq[ievt, 3]
            q2 = mu2_pq[ievt, 3]
            if np.isfinite(q1) and np.isfinite(q2):
                track_vars_out["mu1_px"].append([float(mu1_pq[ievt, 0])])
                track_vars_out["mu1_py"].append([float(mu1_pq[ievt, 1])])
                track_vars_out["mu1_pz"].append([float(mu1_pq[ievt, 2])])
                track_vars_out["mu1_q"].append([int(round(float(q1)))])

                track_vars_out["mu2_px"].append([float(mu2_pq[ievt, 0])])
                track_vars_out["mu2_py"].append([float(mu2_pq[ievt, 1])])
                track_vars_out["mu2_pz"].append([float(mu2_pq[ievt, 2])])
                track_vars_out["mu2_q"].append([int(round(float(q2)))])
            else:
                track_vars_out["mu1_px"].append([]); track_vars_out["mu1_py"].append([]); track_vars_out["mu1_pz"].append([]); track_vars_out["mu1_q"].append([])
                track_vars_out["mu2_px"].append([]); track_vars_out["mu2_py"].append([]); track_vars_out["mu2_pz"].append([]); track_vars_out["mu2_q"].append([])

            dimu_valid += 1
        else:
            track_vars_out["dimu_vx"].append([]); track_vars_out["dimu_vy"].append([]); track_vars_out["dimu_vz"].append([])
            track_vars_out["dimu_px"].append([]); track_vars_out["dimu_py"].append([]); track_vars_out["dimu_pz"].append([])
            track_vars_out["dimu_mass"].append([]); track_vars_out["dimu_chi2"].append([])
            track_vars_out["dimu_xF"].append([]); track_vars_out["dimu_x1"].append([]); track_vars_out["dimu_x2"].append([])
            track_vars_out["dimu_pT"].append([]); track_vars_out["dimu_phi"].append([]); track_vars_out["dimu_costheta"].append([])
            track_vars_out["mu1_px"].append([]); track_vars_out["mu1_py"].append([]); track_vars_out["mu1_pz"].append([]); track_vars_out["mu1_q"].append([])
            track_vars_out["mu2_px"].append([]); track_vars_out["mu2_py"].append([]); track_vars_out["mu2_pz"].append([]); track_vars_out["mu2_q"].append([])
    else:
        track_vars_out["dimu_vx"].append([]); track_vars_out["dimu_vy"].append([]); track_vars_out["dimu_vz"].append([])
        track_vars_out["dimu_px"].append([]); track_vars_out["dimu_py"].append([]); track_vars_out["dimu_pz"].append([])
        track_vars_out["dimu_mass"].append([]); track_vars_out["dimu_chi2"].append([])
        track_vars_out["dimu_xF"].append([]); track_vars_out["dimu_x1"].append([]); track_vars_out["dimu_x2"].append([])
        track_vars_out["dimu_pT"].append([]); track_vars_out["dimu_phi"].append([]); track_vars_out["dimu_costheta"].append([])
        track_vars_out["mu1_px"].append([]); track_vars_out["mu1_py"].append([]); track_vars_out["mu1_pz"].append([]); track_vars_out["mu1_q"].append([])
        track_vars_out["mu2_px"].append([]); track_vars_out["mu2_py"].append([]); track_vars_out["mu2_pz"].append([]); track_vars_out["mu2_q"].append([])

    if len(track_vars_out["tx"][-1]) and len(track_vars_out["ty"][-1]):
        both_valid += 1

print(f"[GLB] selection summary: XZ valid={xz_valid}, YZ valid={yz_valid}, Both={both_valid} (out of {n_evt_after})")
if dimu_enabled:
    print(f"[DIMU] valid dimuons={dimu_valid}/{n_evt_after} (mass window {dimu_mass_min}..{dimu_mass_max} GeV)")

# ------------------------------ write out ------------------------------
write_rus_file(output_file, event_vars, hit_vars, track_vars_out)
print(f"Finished. Results written to {output_file}")