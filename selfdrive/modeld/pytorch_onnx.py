#!/usr/bin/env python3
"""
driving_policy_nav_demo.py – pure‑nav variant (batch fixed to 1)
==============================================================
This version hard‑codes **batch = 1** so the tensors match the
`1×256` and `1×150` shapes you see flowing into the Concat node.

Key points
----------
* Only two inputs:
    • `nav_features` → tensor shape **(1, 256)**
    • `nav_instructions` → tensor shape **(1, 150)**
* Navigation branch: two residual MLP blocks → (1, 256).
* Concatenate `(1,256)` + `(1,150)` → `(1,406)` → lightweight policy head.
* Exports a fixed‑batch ONNX (no dynamic axes) and checks it with ORT.

Dependencies::

    pip install torch onnx onnxruntime numpy
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort

# ---------------------------------------------------------------------------
# 1 · Helper – build the 150‑D instruction strip
# ---------------------------------------------------------------------------

def build_nav_instructions(
    maneuvers: list[dict[str, float | str]],
    *,
    width: int = 150,
    bucket_m: float = 20.0,
    center: int = 25,
) -> np.ndarray:
    """Return (1, 150) float32 strip built from distance / modifier pairs."""
    strip = np.zeros((1, width), dtype=np.float32)
    for m in maneuvers:
        d_idx = center + int(m["distance"] / bucket_m)
        if not 0 <= d_idx < width // 3:
            continue
        dir_idx = 0  # straight
        mod = str(m["modifier"]).lower()
        if mod in ("left", "slight left", "sharp left"):
            dir_idx = 1
        elif mod in ("right", "slight right", "sharp right"):
            dir_idx = 2
        strip[0, d_idx * 3 + dir_idx] = 1.0
    return strip

# ---------------------------------------------------------------------------
# 2 · Navigation branch – residual MLP (256 → 512 → 256) ×2
# ---------------------------------------------------------------------------

class NavBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.project = nn.Linear(256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (1,256)
        h = F.relu(self.fc1(x))
        h = self.fc2(h) + x          # residual 1
        h = F.relu(h)
        h2 = F.relu(self.fc3(h))
        h = self.fc4(h2) + h         # residual 2
        h = F.relu(h)
        return F.relu(self.project(h))

# ---------------------------------------------------------------------------
# 3 · Policy head (concat 256+150 → 406) → simple MLP
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.nav_branch = NavBranch()
        self.head = nn.Sequential(
            nn.Linear(256 + 150, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),  # extend / replace as needed
        )

    def forward(self, nav_feat: torch.Tensor, nav_instr: torch.Tensor) -> torch.Tensor:
        nav_vec = self.nav_branch(nav_feat)            # (1,256)
        fused = torch.cat([nav_vec, nav_instr], dim=1) # (1,406)
        return self.head(fused)                        # (1,128)

# ---------------------------------------------------------------------------
# 4 · Demo – build inputs, export ONNX, verify with ORT
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Build single‑row tensors ----------------------------------------
    nav_features = torch.randn(1, 256, dtype=torch.float32)
    nav_instr_np = build_nav_instructions([
        {"distance": 60.0, "modifier": "right"},
        {"distance": -40.0, "modifier": "left"},
    ])
    nav_instructions = torch.from_numpy(nav_instr_np)

    model = PolicyNet().eval()

    with torch.no_grad():
        torch_out = model(nav_features, nav_instructions)
    print("PyTorch output shape:", torch_out.shape)

    # --- Export fixed‑batch ONNX (no dynamic axes) -----------------------
    onnx_path = "policy_nav_only_fixedB.onnx"
    torch.onnx.export(
        model,
        (nav_features, nav_instructions),
        onnx_path,
        input_names=["nav_features", "nav_instructions"],
        output_names=["policy_out"],
        opset_version=16,
    )
    print("Exported →", onnx_path)

    # --- quick ORT sanity check ------------------------------------------
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {
        "nav_features": nav_features.numpy(),
        "nav_instructions": nav_instr_np,
    })[0]
    print("ONNX output shape :", ort_out.shape)


if __name__ == "__main__":
    main()
