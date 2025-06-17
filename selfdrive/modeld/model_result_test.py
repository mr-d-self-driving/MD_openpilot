#!/usr/bin/env python3
# compare_policy_only.py
"""
Compare PyTorch policy output with the golden tensor saved in each .npz frame.

Recording requirements
----------------------
Each inputs_XXXXX.npz file must contain:
  • desire                       f16
  • traffic_convention           f16
  • lateral_control_params       f16
  • features_buffer              f16
  • prev_desired_curv            f16
  • nav_features                 f16
  • nav_instructions             f16
  • policy_out                   f16   <-- golden reference

Usage
-----
  python compare_policy_only.py <recording_dir>
"""

from pathlib import Path
import sys, time, numpy as np
import onnx
from onnx2pytorch import ConvertModel
import torch

# --------------------------------------------------------------------------- #
#                            --- CONFIG ---                                   #
# --------------------------------------------------------------------------- #
path_to_onnx_policy_model = (
    "/home/adas/openpilot/selfdrive/modeld/models/driving_policy_with_nav_ir10.onnx"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATOL, RTOL = 1e-5, 1e-3

# --------------------------------------------------------------------------- #
#                      --- load policy as PyTorch ---                         #
# --------------------------------------------------------------------------- #
path_to_chkpt = Path(
    "/home/adas/openpilot/selfdrive/modeld/models/driving_policy_pt.pth"
)

print("• Building blank policy and loading state-dict …")
torch_policy = ConvertModel(onnx.load(path_to_onnx_policy_model)).to(device)
torch_policy.load_state_dict(torch.load(path_to_chkpt, map_location=device))
torch_policy.requires_grad_(False)
torch_policy.eval()

# --------------------------------------------------------------------------- #
#                           --- helpers ---                                   #
# --------------------------------------------------------------------------- #
def to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float16, copy=False)).to(device)

def iterate_frames(folder: Path):
    for f in sorted(folder.glob("inputs_*.npz")):
        yield int(f.stem.split("_")[-1]), np.load(f)

# --------------------------------------------------------------------------- #
#                               --- main ---                                  #
# --------------------------------------------------------------------------- #
def main(record_dir: Path):
    if not record_dir.is_dir():
        sys.exit(f"[error] Folder {record_dir} not found")

    mismatches = 0
    start = time.time()

    # Keys required for policy feed
    pol_keys = (
        "desire",
        "traffic_convention",
        "lateral_control_params",
        "features_buffer",
        "prev_desired_curv",
        "nav_features",
        "nav_instructions",
    )

    for step, frame in iterate_frames(record_dir):
        # ------------ build torch inputs -------------
        missing = [k for k in pol_keys if k not in frame]
        if missing:
            sys.exit(f"[error] Frame {step:06d} missing {missing}")

        torch_feed = {k: to_tensor(frame[k]) for k in pol_keys}

        with torch.inference_mode():
            torch_out = torch_policy(**torch_feed)[0].detach().cpu().numpy()

        if "policy_out" not in frame:
            sys.exit(f"[error] Frame {step:06d} lacks 'policy_out'")

        golden = frame["policy_out"]

        if not np.allclose(torch_out, golden, rtol=RTOL, atol=ATOL):
            diff = np.average(np.abs(torch_out - golden))
            print(f"[warn] Frame {step:06d}: drift |max diff| = {diff:.3e}")
            mismatches += 1

    elapsed = time.time() - start
    print(
        f"\nFinished {step+1} frames in {elapsed:.2f}s  "
        f"({(step+1)/elapsed:.1f} FPS)"
    )
    if mismatches:
        print(f"[result] {mismatches} frame(s) failed")
        sys.exit(1)
    else:
        print("[result] All policy outputs match  ✔")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python compare_policy_only.py <recording_dir>")
    main(Path(sys.argv[1]))
