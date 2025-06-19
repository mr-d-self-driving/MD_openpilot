#!/usr/bin/env python3
"""
onxx_single_frame_infer.py
==========================
Run the nav‑aware OpenPilot driving‑policy model **once** and print both the
5884‑D prediction vector and the ground‑truth target.

✔ Accepts a `--frame_idx` (default 0) so you can step through the replay log.
✔ Automatically fabricates any missing tensors using real shapes from the
  first frame or reasonable fall‑backs (unknown dims → 4).

Example
-------
```bash
python onxx_single_frame_infer.py \
       --logdir /home/adas/openpilot/policy_logger \
       --onnx   /home/adas/openpilot/selfdrive/modeld/models/driving_policy_with_nav.onnx \
       --frame_idx 42 --device cpu
```
"""
from __future__ import annotations

import argparse
import warnings
from collections import defaultdict, deque
from pathlib import Path

import onnx
import torch
from onnx2pytorch import ConvertModel
from torch.utils.data import Dataset
import torch.nn.functional as F                 # add near other imports


warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
P = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
P.add_argument("--logdir", type=Path, default='/home/adas/openpilot/policy_logger', help="Folder with batch_*.pt shards from policy_logger")
P.add_argument("--onnx",  type=Path, default='/home/adas/openpilot/selfdrive/modeld/models/driving_policy_with_nav.onnx', help="Nav‑aware ONNX model")
P.add_argument("--frame_idx", type=int, default=0, help="Which frame to run (0‑based index)")
P.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
args = P.parse_args()

DEVICE = torch.device(args.device)
print(f"▶  Device: {DEVICE} | Frame index: {args.frame_idx}")


class PolicyReplay(Dataset):
    def __init__(self, shard_dir: Path):
        self.files = sorted(shard_dir.glob("batch_*.pt"))
        self.index: list[tuple[int, int]] = []
        for f_idx, f in enumerate(self.files):
            num = len(torch.load(f, map_location="cpu"))
            self.index.extend([(f_idx, i) for i in range(num)])
        print(f"✔  Replay contains {len(self.index):,} frames across {len(self.files)} shards")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f_idx, rec_idx = self.index[idx]
        rec = torch.load(self.files[f_idx], map_location="cpu")[rec_idx]
        return rec["in"], rec["out"].to(torch.float32)

replay = PolicyReplay(args.logdir)
frame_idx = args.frame_idx % len(replay)
inputs_raw, target = replay[frame_idx]
print(f"▶  Loaded frame {frame_idx} from log")

# ---------------------------------------------------------------------------
# ONNX → PyTorch conversion
# ---------------------------------------------------------------------------
print(f"▶  Loading & converting ONNX model: {args.onnx}")
onnx_model = onnx.load(str(args.onnx))
policy_pt = ConvertModel(onnx_model).to(DEVICE).eval()
onnx_inputs = [t.name for t in onnx_model.graph.input]
print(f"✔  Model expects {len(onnx_inputs)} input tensors")


missing = [n for n in onnx_inputs if n not in inputs_raw]
if missing:
    print("\nInputs expected by the ONNX graph but **absent** in this frame:")
    for n in missing:
        proto = onnx_model.graph.input[onnx_inputs.index(n)].type.tensor_type
        dims = [d.dim_value if d.dim_value > 0 else "?" for d in proto.shape.dim]
        print(f" • {n:<18} shape={dims}")
else:
    print("\nAll ONNX inputs satisfied by this frame ✓")
print("────────────────────────────────────────────────────────────\n")


ref_shapes = {k: tuple(v.shape) for k, v in inputs_raw.items()}


inputs_raw = {
    k: (torch.as_tensor(v, device=DEVICE)          # NumPy → Tensor
        if not torch.is_tensor(v) else             # already Tensor?
        v.to(DEVICE, non_blocking=True))           # Tensor → CUDA
    for k, v in inputs_raw.items()
}
target = torch.tensor(target).to(DEVICE)
with torch.no_grad():
    preds = policy_pt(**inputs_raw)  # drop batch dim
    loss  = F.mse_loss(preds, target)

print(f"\nMSE loss    → {loss.item():.6f}")

