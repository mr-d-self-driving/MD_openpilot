#!/usr/bin/env python3
"""
inject_nav_inputs.py
--------------------
Adds two new inputs (nav_features, nav_instructions) to an existing Gemm
by concatenating them with the old activation.  Strategy A from the table.

Result: weight matrix grows from (256, 512)  →  (256, 918).

⚠️  The newly-added columns are zero-initialised; fine-tune the
    model afterwards so it learns meaningful weights.
"""
from pathlib import Path
import numpy as np
import onnx
import onnx_graphsurgeon as gs

IN_FILE  = "/home/adas/openpilot/selfdrive/modeld/models/driving_policy.onnx"
OUT_FILE = "/home/adas/openpilot/selfdrive/modeld/models/driving_policy_with_nav.onnx"

print(f"Loading {IN_FILE} …")
onnx_model = onnx.load(IN_FILE)
graph = gs.import_onnx(onnx_model)

# ------------------------------------------------------------------ #
# 1.  Create two new graph inputs
# ------------------------------------------------------------------ #
nav_feat  = gs.Variable("nav_features",  dtype=np.float16, shape=(1, 256))
nav_instr = gs.Variable("nav_instructions", dtype=np.float16, shape=(1, 150))
graph.inputs.extend([nav_feat, nav_instr])

# ------------------------------------------------------------------ #
# 2.  Locate the *old* Gemm node (index 81 in your dump)
# ------------------------------------------------------------------ #
gemm = graph.nodes[81]                     # or search by name/path if safer
assert gemm.op == "Gemm", "Node 81 is not Gemm!"

old_act   = gemm.inputs[0]                 # shape (1, 512)
old_W     = gemm.inputs[1]                 # Constant tensor (256, 512)
bias      = gemm.inputs[2]                 # Constant tensor (256)

# ------------------------------------------------------------------ #
# 3.  Build Concat(axis=1) → new activation
# ------------------------------------------------------------------ #
concat_out = gs.Variable(
    "gemm_concat_input",
    dtype=np.float16,
    shape=(1, 512 + 256 + 150)       # 918
)

concat = gs.Node(
    op="Concat",
    inputs=[old_act, nav_feat, nav_instr],
    outputs=[concat_out],
    attrs={"axis": 1},
)
graph.nodes.append(concat)

# Plug Concat’s output into Gemm
gemm.inputs[0] = concat_out

# ------------------------------------------------------------------ #
# 4.  Enlarge the weight matrix to (256, 918)
# ------------------------------------------------------------------ #
old_W_np = old_W.values                    # ndarray (256, 512)
pad_nav_feat   = np.zeros((256, 256), dtype=np.float16)
pad_nav_instr  = np.zeros((256, 150), dtype=np.float16)
new_W_np = np.concatenate(
    [old_W_np, pad_nav_feat, pad_nav_instr], axis=1
)                                           # (256, 918)

new_W = gs.Constant(
    "policy_model.temporal_hydra.in_layer.plan.weight_extended",
    values=new_W_np,
)
gemm.inputs[1] = new_W                     # replace weight

# bias stays the same (length 256)

# ------------------------------------------------------------------ #
# 5.  Clean up and save
# ------------------------------------------------------------------ #
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), OUT_FILE)
print(f"Saved patched model →  {OUT_FILE}")
