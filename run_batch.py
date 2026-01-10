import sys, subprocess, shutil
from pathlib import Path
from copy import deepcopy
import yaml

ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "config" / "settings.yaml"

def run_one(cfg_dict):
    text = yaml.safe_dump(cfg_dict, sort_keys=False)
    CFG_PATH.write_text(text)
    print("\n=== running main.py with ===")
    print(f"input_file : {cfg_dict.get('input_file')}")
    print(f"output_file: {cfg_dict.get('output_file')}\n")
    subprocess.run([sys.executable, str(ROOT / "main.py")], check=True)

def main():
    if not CFG_PATH.exists():
        print(f"ERROR: {CFG_PATH} not found.", file=sys.stderr)
        sys.exit(1)
    original = CFG_PATH.read_text()
    try:
        base_cfg = yaml.safe_load(original) or {}
        jobs = base_cfg.pop("batch_jobs", None)
        if not jobs:
            print("No 'batch_jobs' found; forwarding current settings.yaml to main.py once.")
            run_one(base_cfg)
            return
        base_wo_io = deepcopy(base_cfg)
        base_wo_io.pop("input_file", None)
        base_wo_io.pop("output_file", None)
        for i, job in enumerate(jobs, 1):
            job_in = job.get("input")
            job_out = job.get("output")
            if not job_in or not job_out:
                print(f"WARNING: job #{i} missing input/output → skipping")
                continue
            cfg_i = deepcopy(base_wo_io)
            cfg_i["input_file"] = job_in
            cfg_i["output_file"] = job_out
            run_one(cfg_i)
        print("\nBatch complete.")
    finally:
        CFG_PATH.write_text(original)

if __name__ == "__main__":
    main()
