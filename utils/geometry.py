import numpy as np
import pandas as pd

_NDET = 64

_FILE_COLS = [
    "ipl","z","nelem","cellwidth","spacing","xoffset",
    "scalex","x0","x1","x2","costheta","scaley","y0","y1","y2","sintheta",
    "resolution","p1x_w1","p1y_w1","p1z_w1","deltapx","deltapy","deltapz",
    "dp1x","dp1y","dp1z"
]

_GPLANE_DTYPE = np.dtype([
    ("z",np.float32,(_NDET,)),
    ("nelem",np.int32,(_NDET,)),
    ("cellwidth",np.float32,(_NDET,)),
    ("spacing",np.float32,(_NDET,)),
    ("xoffset",np.float32,(_NDET,)),
    ("scalex",np.float32,(_NDET,)),
    ("x0",np.float32,(_NDET,)),
    ("x1",np.float32,(_NDET,)),
    ("x2",np.float32,(_NDET,)),
    ("costheta",np.float32,(_NDET,)),
    ("scaley",np.float32,(_NDET,)),
    ("y0",np.float32,(_NDET,)),
    ("y1",np.float32,(_NDET,)),
    ("y2",np.float32,(_NDET,)),
    ("sintheta",np.float32,(_NDET,)),
    ("resolution",np.float32,(_NDET,)),
    ("p1x_w1",np.float32,(_NDET,)),
    ("p1y_w1",np.float32,(_NDET,)),
    ("p1z_w1",np.float32,(_NDET,)),
    ("deltapx",np.float32,(_NDET,)),
    ("deltapy",np.float32,(_NDET,)),
    ("deltapz",np.float32,(_NDET,)),
    ("dp1x",np.float32,(_NDET,)),
    ("dp1y",np.float32,(_NDET,)),
    ("dp1z",np.float32,(_NDET,)),
    ("slope_max",np.float32,(_NDET,)),
    ("inter_max",np.float32,(_NDET,))
])

def load_gplane(path: str) -> np.ndarray:
    df = pd.read_csv(path, delim_whitespace=True, comment="#", header=None, names=_FILE_COLS)
    gp = np.zeros((), dtype=_GPLANE_DTYPE)
    for _, r in df.iterrows():
        i = int(r["ipl"])
        if not (0 <= i < _NDET):
            continue
        gp["z"][i] = float(r["z"])
        gp["nelem"][i] = int(float(r["nelem"]))
        gp["cellwidth"][i] = float(r["cellwidth"])
        gp["spacing"][i] = float(r["spacing"])
        gp["xoffset"][i] = float(r["xoffset"])
        gp["scalex"][i] = float(r["scalex"])
        gp["x0"][i] = float(r["x0"])
        gp["x1"][i] = float(r["x1"])
        gp["x2"][i] = float(r["x2"])
        gp["costheta"][i] = float(r["costheta"])
        gp["scaley"][i] = float(r["scaley"])
        gp["y0"][i] = float(r["y0"])
        gp["y1"][i] = float(r["y1"])
        gp["y2"][i] = float(r["y2"])
        gp["sintheta"][i] = float(r["sintheta"])
        gp["resolution"][i] = float(r["resolution"])
        gp["p1x_w1"][i] = float(r["p1x_w1"])
        gp["p1y_w1"][i] = float(r["p1y_w1"])
        gp["p1z_w1"][i] = float(r["p1z_w1"])
        gp["deltapx"][i] = float(r["deltapx"])
        gp["deltapy"][i] = float(r["deltapy"])
        gp["deltapz"][i] = float(r["deltapz"])
        gp["dp1x"][i] = float(r["dp1x"])
        gp["dp1y"][i] = float(r["dp1y"])
        gp["dp1z"][i] = float(r["dp1z"])
    def _dbg(i):
        sp_cm = gp["spacing"][i] * 0.1
        xoff_cm = gp["xoffset"][i] * 0.1
        return i, float(gp["z"][i]), int(gp["nelem"][i]), sp_cm, xoff_cm
    for det in (15, 16, 21, 22, 27, 28):
        i, z, ne, sp_cm, xoff_cm = _dbg(det)
        print(f"[plane] det={i:2d} z={z:.3f}cm ne={ne} sp={sp_cm:.4f}cm xoff={xoff_cm:.3f}cm")
    blob_u8 = gp.tobytes(order="C")
    blob_f32 = np.frombuffer(blob_u8, dtype=np.float32)
    return blob_f32
