#!/usr/bin/env python3
"""
inject_nav_inputs_parallel.py
=============================
Strategy B: keep the original Gemm (backbone) and add two
parallel Gemms (nav_features, nav_instructions); sum the three
outputs with an Add.

After running, the model has three external inputs:

  • /temporal_hydra/resblock/final_relu/Relu_output_0   (1×512 FP16)
  • nav_features                                        (1×256 FP16)
  • nav_instructions                                    (1×150 FP16)

and still produces the same 256-D plan activation, now as
/temporal_hydra/plan/sum_out.

FINETUNING REQUIRED – the new weights/biases start at zero.
"""
from pathlib import Path
import numpy as np
import onnx
import onnx_graphsurgeon as gs

# -------------------------------------------------------------------- #
# Config
# -------------------------------------------------------------------- #
IN_FILE  = "/home/adas/openpilot/selfdrive/modeld/models/driving_policy.onnx"
OUT_FILE = "/home/adas/openpilot/selfdrive/modeld/models/driving_policy_with_nav_parallel.onnx"

# -------------------------------------------------------------------- #
# Helper
# -------------------------------------------------------------------- #
def zeros_fp16(shape):
    return np.zeros(shape, dtype=np.float16)

# -------------------------------------------------------------------- #
# 0.  Load + import into graphsurgeon
# -------------------------------------------------------------------- #
print(f"Loading {IN_FILE} …")
model = onnx.load(IN_FILE)
graph = gs.import_onnx(model)

# -------------------------------------------------------------------- #
# 1.  Find the ORIGINAL Gemm (index 81 in your dump)
# -------------------------------------------------------------------- #
try:
    gemm0 = graph.nodes[81]
    assert gemm0.op == "Gemm"
except (IndexError, AssertionError):
    raise RuntimeError("Node 81 is not a Gemm — adjust the index or search by name.")

backbone_act  = gemm0.inputs[0]      # (1,512)
W0, b0        = gemm0.inputs[1:3]    # existing constants
gemm0_out     = gemm0.outputs[0]     # (1,256)

# -------------------------------------------------------------------- #
# 2.  Create two new external inputs
# -------------------------------------------------------------------- #
nav_feat   = gs.Variable("nav_features",      dtype=np.float16, shape=(1, 256))
nav_instr  = gs.Variable("nav_instructions",  dtype=np.float16, shape=(1, 150))
graph.inputs.extend([nav_feat, nav_instr])

# -------------------------------------------------------------------- #
# 3.  Gemm₁  (nav_features → 256)
# -------------------------------------------------------------------- #
W1 = gs.Constant("nav_feat_weight",  values=zeros_fp16((256, 256)))
b1 = gs.Constant("nav_feat_bias",    values=zeros_fp16((256,)))
gemm1_out = gs.Variable()

gemm1 = gs.Node(
    op="Gemm",
    name="Gemm_nav_features",
    inputs=[nav_feat, W1, b1],
    outputs=[gemm1_out],
    attrs={"alpha": 1.0, "beta": 1.0, "transB": 1},
)
graph.nodes.append(gemm1)

# -------------------------------------------------------------------- #
# 4.  Gemm₂  (nav_instructions → 256)
# -------------------------------------------------------------------- #
W2 = gs.Constant("nav_instr_weight", values=zeros_fp16((256, 150)))
b2 = gs.Constant("nav_instr_bias",   values=zeros_fp16((256,)))
gemm2_out = gs.Variable()

gemm2 = gs.Node(
    op="Gemm",
    name="Gemm_nav_instructions",
    inputs=[nav_instr, W2, b2],
    outputs=[gemm2_out],
    attrs={"alpha": 1.0, "beta": 1.0, "transB": 1},
)
graph.nodes.append(gemm2)

# -------------------------------------------------------------------- #
# 5.  Sum outputs with Add  (NEW tensor name!)
# -------------------------------------------------------------------- #
sum_out = gs.Variable(
    "/temporal_hydra/plan/sum_out",
    dtype=np.float16,
    shape=(1, 256),
)

add_node = gs.Node(
    op="Add",
    name="Add_plan_sum",
    inputs=[gemm0_out, gemm1_out, gemm2_out],
    outputs=[sum_out],
)
graph.nodes.append(add_node)

# -------------------------------------------------------------------- #
# 6.  Redirect downstream consumers of gemm0_out → sum_out
# -------------------------------------------------------------------- #
for node in graph.nodes:
    for idx, inp in enumerate(node.inputs):
        if inp is gemm0_out and node is not add_node:
            node.inputs[idx] = sum_out

# -------------------------------------------------------------------- #
# 7.  Clean up & save
# -------------------------------------------------------------------- #
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), OUT_FILE)
print(f"✅  Saved patched model →  {OUT_FILE}")
print("⚠️  Remember to fine-tune the new weights before deployment!")
