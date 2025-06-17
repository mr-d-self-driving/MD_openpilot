#!/usr/bin/env python3
# model_test_load.py – robust checkpoint loader
# ------------------------------------------------------------
import inspect, sys
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as rt
from onnx2pytorch import ConvertModel

# ── paths ────────────────────────────────────────────────────
ROOT      = Path("/home/adas/openpilot/selfdrive/modeld/models")
ONNX_PATH = ROOT / "driving_policy_with_nav_ir10.onnx"
CKPT_PATH = ROOT / "driving_policy_state_dict_ir10.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float16

# ─────────────────────────────────────────────────────────────
# 1. Build the architecture                                   |
#    ▸ try to keep initializers as Parameters if the           |
#      installed onnx2pytorch supports it                      |
# ─────────────────────────────────────────────────────────────
def build_model():
    kwargs = {}
    sig = inspect.signature(ConvertModel.__init__)
    if "keep_initializers_as_params" in sig.parameters:
        kwargs["keep_initializers_as_params"] = True   # old-checkpoint-compatible
        print("[i] using keep_initializers_as_params=True")
    else:
        print("[i] parameter not supported (older onnx2pytorch) – falling back")

    model = ConvertModel(onnx.load(ONNX_PATH), **kwargs).to(device)
    model = model.half()               # fp16 like the checkpoint
    model.requires_grad_(False).eval()
    return model

# ─────────────────────────────────────────────────────────────
# 2. (Optional) create checkpoint if it doesn’t exist          |
# ─────────────────────────────────────────────────────────────
if not CKPT_PATH.exists():
    print("[+] Converting ONNX → PyTorch for the FIRST time; saving weights …")
    tmp = build_model()
    torch.save(tmp.state_dict(), CKPT_PATH,_use_new_zipfile_serialization=True)
    del tmp                                  # free GPU
else:
    print("[i] Found existing checkpoint:", CKPT_PATH)

# ─────────────────────────────────────────────────────────────
# 3. Rebuild architecture & load weights                      |
#    ▸ filter out keys that no longer exist                   |
# ─────────────────────────────────────────────────────────────
print("[+] Re-creating module and restoring weights …")
model = build_model()
saved_sd   = torch.load(CKPT_PATH, map_location=device,weights_only=True)
model_sd   = model.state_dict()

# keep only the intersection
filtered_sd = {k: v for k, v in saved_sd.items() if k in model_sd}
unused_keys = [k for k in saved_sd.keys() if k not in model_sd]
missing     = [k for k in model_sd.keys() if k not in filtered_sd]

model.load_state_dict(filtered_sd, strict=False)
print(f"    ✓ loaded {len(filtered_sd)} params "
      f"({len(unused_keys)} ignored, {len(missing)} missing)")

# ─────────────────────────────────────────────────────────────
# 4. Quick numerical sanity check vs. ONNXRuntime             |
# ─────────────────────────────────────────────────────────────
print("[+] Running fp16 sanity check …")
torch_inputs = {
    'desire':               torch.ones((1,25,8),   dtype=dtype, device=device),
    'traffic_convention':   torch.ones((1,2),      dtype=dtype, device=device),
    'lateral_control_params':torch.ones((1,2),     dtype=dtype, device=device),
    'prev_desired_curv':    torch.ones((1,25,1),   dtype=dtype, device=device),
    'features_buffer':      torch.ones((1,25,512), dtype=dtype, device=device),
    'nav_features':         torch.ones((1,256),    dtype=dtype, device=device),
    'nav_instructions':     torch.ones((1,150),    dtype=dtype, device=device),
}
onnx_inputs = {k: v.cpu().numpy() for k, v in torch_inputs.items()}

with torch.inference_mode():
    torch_out = model(**torch_inputs).cpu().numpy()

sess = rt.InferenceSession(str(ONNX_PATH),
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
onnx_out, = sess.run(None, onnx_inputs)

diff = np.mean(np.abs(torch_out.astype(np.float32) -
                      onnx_out.astype(np.float32)))
print(f"    output shape : {torch_out.shape}")
print(f"    mean |Δ|     : {diff:.3e}")
print("[✓] finished")
