# main.py — ER -> XZ -> YZ (per XZ) -> Global combine -> write
import yaml
import awkward as ak
import numpy as np
from utils.reader import read_rus_file
from utils.writer import write_rus_file
from utils.geometry import load_gplane
from utils.gpu_kernels import (
    run_event_reduction,
    run_xz_tracking,
    run_yz_tracking,
    run_global_combine,
    MAX_HITS_ER,
)

# hit feature indices
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

# global KMAG anchor
z_kmag_bend = glb_cfg.get("z_kmag_bend_cm", None)
if z_kmag_bend is None:
    z_kmag_bend = float("nan")
    print("[global] KMAG anchor: DISABLED (z_kmag_bend_cm is null/missing)")
else:
    z_kmag_bend = float(z_kmag_bend)
    print(f"[global] KMAG anchor: ENABLED at z = {z_kmag_bend:.3f} cm")

d0_x_ids = glb_cfg.get("d0_x_det_ids", [3, 4])
d0_x_window = float(glb_cfg.get("d0_x_window_cm", 1.5))   # cm
pt_kick_kmag = float(glb_cfg.get("pt_kick_kmag", 0.0075))  # slope scale
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
    hit_vars["hitsInTime"] = ak.Array([[True]*len(ev) for ev in hit_vars["detectorID"]])

# pack ER input (always allocate; ER can be off)
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
pruned = {k: [] for k in hit_vars.keys()}
kept_mask = ~hastoomany
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
        tracks_yz, ntracks_yz = run_yz_tracking(
            hit_vars, gplane_array, hastoomany_trim,
            tracks_xz, ntracks_xz
        )

# ------------------------------ global combine ------------------------------
D0_X_IDS = set(int(d) for d in d0_x_ids)
d0_with_hits = sum(
    any(int(d) in D0_X_IDS for d in hit_vars["detectorID"][i])
    for i in range(len(hit_vars["detectorID"]))
)
print(f"[global] Events with any D0-X hits {sorted(D0_X_IDS)}: "
      f"{d0_with_hits}/{len(hit_vars['detectorID'])}")

tracks_glb, ntracks_glb = run_global_combine(
    tracks_xz if tracks_xz is not None else np.zeros((n_evt_after,1,160), np.float32),
    ntracks_xz if ntracks_xz is not None else np.zeros((n_evt_after,), np.uint32),
    tracks_yz if tracks_yz is not None else np.zeros((n_evt_after,1,160), np.float32),
    ntracks_yz if ntracks_yz is not None else np.zeros((n_evt_after,), np.uint32),
    hit_vars, gplane_array,
    z_ref=z_kmag_bend,
    d0_ids=list(D0_X_IDS), d0_win_cm=d0_x_window, pt_kick=pt_kick_kmag
)

# see how many invP came out finite (slot 14)
if tracks_glb.size:
    invp0 = np.sum(np.isfinite(tracks_glb[:,0,14]) & (tracks_glb[:,0,14] > 0))
    invp1 = np.sum(np.isfinite(tracks_glb[:,1,14]) & (tracks_glb[:,1,14] > 0))
    print(f"[global] invP>0: slot0={invp0}/{tracks_glb.shape[0]} slot1={invp1}/{tracks_glb.shape[0]} (index 14)")
    inwin = np.sum((tracks_glb[:,0,14] > 0.01) & (tracks_glb[:,0,14] < 0.10))
    print(f"[global] invP in [0.01,0.10]: {inwin}/{tracks_glb.shape[0]}")

# ------------------------------ pack combined tree ------------------------------
nan = float("nan")
track_vars_out = {
    "stationID": [],
    "x0": [], "tx": [], "nhits_xz": [], "chi2_xz": [],
    "y0": [], "ty": [], "nhits_yz": [], "chi2_yz": [],
    "invP": []
}

xz_valid = yz_valid = both_valid = 0

for ievt in range(n_evt_after):
    nout = int(ntracks_glb[ievt])

    if nout >= 1 and np.isfinite(tracks_glb[ievt, 0, 3]):
        nhx = int(tracks_glb[ievt,0,2])
        chi2x = float(tracks_glb[ievt,0,3])
        tx = float(tracks_glb[ievt,0,5])
        x0 = float(tracks_glb[ievt,0,7])
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

    if nout >= 2 and np.isfinite(tracks_glb[ievt, 1, 3]):
        nhy = int(tracks_glb[ievt,1,2])
        chi2y = float(tracks_glb[ievt,1,3])
        ty = float(tracks_glb[ievt,1,5])
        y0 = float(tracks_glb[ievt,1,7])
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

    invp0 = tracks_glb[ievt,0,14] if (nout>=1) else nan
    invp1 = tracks_glb[ievt,1,14] if (nout>=2) else nan
    invp = invp0 if np.isfinite(invp0) else (invp1 if np.isfinite(invp1) else nan)
    if np.isfinite(invp):
        track_vars_out["invP"].append([float(invp)])
    else:
        track_vars_out["invP"].append([])

    track_vars_out["stationID"].append([1] if (
        len(track_vars_out["tx"][-1]) or len(track_vars_out["ty"][-1])) else [])

    if (len(track_vars_out["tx"][-1]) and len(track_vars_out["ty"][-1])):
        both_valid += 1

print(f"[GLB] selection summary: XZ valid={xz_valid}, YZ valid={yz_valid}, Both={both_valid} (out of {n_evt_after})")

# ------------------------------ write out ------------------------------
write_rus_file(output_file, event_vars, hit_vars, track_vars_out)
print(f"Finished. Results written to {output_file}")
