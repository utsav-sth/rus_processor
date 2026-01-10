import uproot

def write_rus_file(output_file, event_vars, hit_vars, track_vars):
    with uproot.recreate(output_file) as f:
        all_vars = {**event_vars, **hit_vars, **track_vars}
        f["tree"] = all_vars


