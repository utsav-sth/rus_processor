# utils/reader.py

import uproot
import numpy as np
import awkward as ak

def read_rus_file(file_path):
    with uproot.open(file_path) as f:
        if "tree" not in f:
            raise KeyError("Tree not found in the ROOT file.")
        tree = f["tree"]
        keys = tree.keys()

        event_vars = {}
        scalar_keys = ["runID", "spillID", "eventID", "turnID", "rfID"]
        for key in scalar_keys:
            if key in keys:
                event_vars[key] = tree[key].array(library="np")
            else:
                print(f"Warning: '{key}' not found in tree — skipped.")

        array_keys = ["fpgaTrigger", "nimTrigger", "rfIntensity"]
        for key in array_keys:
            if key in keys:
                event_vars[key] = ak.to_numpy(tree[key].array())
            else:
                print(f"Warning: '{key}' not found in tree — skipped.")

        hit_vars = {}
        hit_keys = ["detectorID", "elementID", "tdcTime", "driftDistance", "hitsInTime"]
        for key in hit_keys:
            if key in keys:
                hit_vars[key] = tree[key].array()
            else:
                print(f"Warning: '{key}' not found in tree - skipped.")

        #Fall back, fabricate hitsInTime (all-true) if the branch is absent
        if "hitsInTime" not in hit_vars:
            if "detectorID" in hit_vars:
                det_jagged = hit_vars["detectorID"]
                hit_vars["hitsInTime"] = ak.Array(
                    [[True] * len(evt) for evt in det_jagged]
                )
                print("Info: fabricated default hitsInTime=True for every hit.")
            else:
                print("Warning: detectorID missing; cannot create hitsInTime fallback.")

        track_vars = {}
        track_keys = ["hitID", "hitTrackID"]
        for key in track_keys:
            if key in keys:
                track_vars[key] = tree[key].array()
            else:
                print(f"Warning: '{key}' not found in tree - skipped.")

        return event_vars, hit_vars, track_vars



