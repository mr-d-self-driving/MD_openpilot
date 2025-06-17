#!/usr/bin/env python3
"""
inject_nav_inputs.py
====================
Patch two places inside an existing OpenPilot *driving_policy.onnx* so
that navigation information is fed into both the *plan* and *action* heads.

Added components
----------------
1. **NavBranch** — a two‑block residual MLP (all‑zeros initialisation)
   * Input  : `nav_features` **(1,256)** `float16`
   * Output : `nav_vec_relu` **(1,256)**
2. **Concat‑A** — `[resblock/final_relu , nav_vec_relu , nav_instructions]` →
   feeds **`temporal_hydra/plan/Gemm`**
3. **Concat‑B** — `[existing Concat_1 inputs , nav_vec_relu , nav_instructions]`
   so that **`/action_block/.../Gemm`** also sees the +406 new features.
4. Both Gemm weight matrices are right‑padded with zeros to accommodate
   the +406 columns (no change to biases).

After running, the model file `driving_policy_with_nav.onnx` will be ready
for fine‑tuning so those blank weights learn useful behaviour.

Requirements
------------
```bash
pip install onnx==1.15 onnx-graphsurgeon numpy onnxruntime
```
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
import onnx_graphsurgeon as gs

# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
IN_FILE  = Path("/home/adas/openpilot/selfdrive/modeld/models/driving_policy.onnx")
OUT_FILE = IN_FILE.with_name("driving_policy_with_nav.onnx")

print(f"Loading {IN_FILE} …")
onnx_model = onnx.load(str(IN_FILE))
graph = gs.import_onnx(onnx_model)

# --------------------------------------------------------------------------- #
# 1 · Add new graph inputs                                                    #
# --------------------------------------------------------------------------- #
nav_feat  = gs.Variable("nav_features",  dtype=np.float16, shape=(1, 256))
nav_instr = gs.Variable("nav_instructions", dtype=np.float16, shape=(1, 150))

# Extend graph inputs idempotently
for var in (nav_feat, nav_instr):
    if var.name not in {i.name for i in graph.inputs}:
        graph.inputs.append(var)

# --------------------------------------------------------------------------- #
# 2 · Helper → zero‑initialised Constant                                      #
# --------------------------------------------------------------------------- #
def zeros(name: str, shape: tuple[int, ...]) -> gs.Constant:
    """Return a float16 all‑zeros Constant of the given *shape*."""
    return gs.Constant(name, values=np.zeros(shape, dtype=np.float16))

# --------------------------------------------------------------------------- #
# 3 · Build NavBranch (two residual blocks + projection)                      #
# --------------------------------------------------------------------------- #
# nav_features (1,256) → nav_vec_relu (1,256)

def residual(name_prefix: str,
             x: gs.Variable,
             width: int,
             hidden: int) -> gs.Variable:
    """One residual MLP block: x → Relu(Gemm(Relu(Gemm(x)))) + x.

    * x            : (1, width)
    * hidden layer : (1, hidden)
    * output       : (1, width) — same as *x* so the Add is dimension‑wise valid.
    """
    # Gemm #1  (1,width) · (width,hidden) = (1,hidden)
    fc1_out  = gs.Variable(f"{name_prefix}_fc1_out", dtype=np.float16)
    fc1      = gs.Node(
        "Gemm",
        inputs=[x,
                zeros(f"{name_prefix}_W1", (width, hidden)),   # (width,hidden)
                zeros(f"{name_prefix}_b1", (hidden,))],        # (hidden,)
        outputs=[fc1_out]
    )

    relu1_out = gs.Variable(f"{name_prefix}_relu1", dtype=np.float16)
    relu1     = gs.Node("Relu", inputs=[fc1_out], outputs=[relu1_out])

    # Gemm #2  (1,hidden) · (hidden,width) = (1,width)
    fc2_out  = gs.Variable(f"{name_prefix}_fc2_out", dtype=np.float16)
    fc2      = gs.Node(
        "Gemm",
        inputs=[relu1_out,
                zeros(f"{name_prefix}_W2", (hidden, width)),   # (hidden,width)
                zeros(f"{name_prefix}_b2", (width,))],         # (width,)
        outputs=[fc2_out]
    )

    add_out  = gs.Variable(f"{name_prefix}_add", dtype=np.float16)
    add      = gs.Node("Add", inputs=[fc2_out, x], outputs=[add_out])

    relu2_out = gs.Variable(f"{name_prefix}_relu2", dtype=np.float16)
    relu2     = gs.Node("Relu", inputs=[add_out], outputs=[relu2_out])

    graph.nodes.extend([fc1, relu1, fc2, add, relu2])
    return relu2_out

# First residual block
blk1_out = residual("nav_blk1", nav_feat, width=256, hidden=512)
# Second residual block
blk2_out = residual("nav_blk2", blk1_out, width=256, hidden=512)

# Projection to 256‑D + activation
nav_vec = gs.Variable("nav_vec", dtype=np.float16)
proj    = gs.Node("Gemm",
                  inputs=[blk2_out, zeros("nav_proj_W", (256, 256)), zeros("nav_proj_b", (256,))],
                  outputs=[nav_vec])

nav_vec_relu = gs.Variable("nav_vec_relu", dtype=np.float16)
relu_final   = gs.Node("Relu", inputs=[nav_vec], outputs=[nav_vec_relu])

graph.nodes.extend([proj, relu_final])

# --------------------------------------------------------------------------- #
# 4 · Utility to extend a Gemm's weight matrix                                #
# --------------------------------------------------------------------------- #

def pad_weight(old: gs.Constant, pad_cols: int, name: str) -> gs.Constant:
    """Right‑pad an existing (out, in) weight matrix with zeros."""
    old_np: np.ndarray = old.values
    rows, cols = old_np.shape
    pad = np.zeros((rows, pad_cols), dtype=old_np.dtype)
    new_np = np.concatenate([old_np, pad], axis=1)
    return gs.Constant(name, values=new_np)

PAD_COLS = 256 + 150  # nav_vec + nav_instructions

# --------------------------------------------------------------------------- #
# 5 · Patch PLAN head                                                         #
# --------------------------------------------------------------------------- #
plan_gemm = next(n for n in graph.nodes if n.op == "Gemm" and n.name.endswith("/plan/Gemm"))
plan_in   = plan_gemm.inputs[0]

plan_concat_out = gs.Variable("plan_concat_in", dtype=np.float16)
plan_concat     = gs.Node("Concat", inputs=[plan_in, nav_vec_relu, nav_instr], outputs=[plan_concat_out], attrs={"axis": 1})

if plan_concat not in graph.nodes:
    graph.nodes.append(plan_concat)
plan_gemm.inputs[0] = plan_concat_out
plan_gemm.inputs[1] = pad_weight(plan_gemm.inputs[1], PAD_COLS, name="plan_weight_extended")

# --------------------------------------------------------------------------- #
# 6 · Patch ACTION head                                                       #
# --------------------------------------------------------------------------- #
ACTION_GEMM_SUFFIX = "/action_block/action_block_in/action_block_in.0/Gemm"
action_gemm = next(n for n in graph.nodes if n.op == "Gemm" and n.name.endswith(ACTION_GEMM_SUFFIX))

act_in = action_gemm.inputs[0]

action_concat_out = gs.Variable("action_concat_in", dtype=np.float16)
action_concat     = gs.Node("Concat", inputs=[act_in, nav_vec_relu, nav_instr], outputs=[action_concat_out], attrs={"axis": 1})

if action_concat not in graph.nodes:
    graph.nodes.append(action_concat)
action_gemm.inputs[0] = action_concat_out
action_gemm.inputs[1] = pad_weight(action_gemm.inputs[1], PAD_COLS, name="action_weight_extended")

# --------------------------------------------------------------------------- #
# 7 · Fix Add‑broadcast order                                             #
# --------------------------------------------------------------------------- #
# onnx2pytorch expects the *first* input of an Add to have the broadcast‑target
# shape (e.g. (1,10,512)) — otherwise in‑place `+=` fails when it starts from
# a 1‑D bias vector. Swap inputs whenever the first is a rank‑1 constant and
# the second is higher‑rank.
for add_node in [n for n in graph.nodes if n.op == "Add" and len(n.inputs) == 2]:
    a, b = add_node.inputs
    if (isinstance(a, gs.Constant) and a.values.ndim == 1 and
            isinstance(b, gs.Variable) and (b.shape is None or len(b.shape) > 1)):
        add_node.inputs = [b, a]  # put high‑rank tensor first

# --------------------------------------------------------------------------- #
# 8 · Clean‑up & save                                                         #
# --------------------------------------------------------------------------- #

graph.cleanup().toposort()
model_proto = gs.export_onnx(graph)
# Downgrade IR version for older onnxruntime builds (<=1.16)
model_proto.ir_version = 10  # ORT <1.17 supports up to 10 only
onnx.save(model_proto, str(OUT_FILE))
print(f"Saved patched model → {OUT_FILE}")

# --------------------------------------------------------------------------- #
# 9 · Quick sanity‑check (optional)                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("✖ onnxruntime not installed — run `pip install onnxruntime` to enable the test.")

    print("Running a dummy forward‑pass …")
    sess = ort.InferenceSession(str(OUT_FILE), providers=["CPUExecutionProvider"])

    dummy_inputs = {}
    for inp in sess.get_inputs():
        shape = [1 if s is None else s for s in inp.shape]  # replace None with 1
        np_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
        dummy_inputs[inp.name] = np.zeros(shape, dtype=np_dtype)

    outs = sess.run(None, dummy_inputs)
    print(f"✓ Inference OK — {len(outs)} output tensors produced:")
    for out_meta, arr in zip(sess.get_outputs(), outs):
        shape_str = str(tuple(arr.shape))
        print(f"  • {out_meta.name:32s} → shape {shape_str:<20s} dtype {arr.dtype}")
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("✖ onnxruntime not installed — run `pip install onnxruntime` to enable the test.")

    print("Running a dummy forward‑pass …")
    sess = ort.InferenceSession(str(OUT_FILE), providers=["CPUExecutionProvider"])

    dummy_inputs = {}
    for inp in sess.get_inputs():
        shape = [1 if s is None else s for s in inp.shape]  # replace None with 1
        np_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
        dummy_inputs[inp.name] = np.zeros(shape, dtype=np_dtype)

    outs = sess.run(None, dummy_inputs)
    print(f"✓ Inference OK — {len(outs)} output tensors produced:")
    for out_meta, arr in zip(sess.get_outputs(), outs):
        shape_str = str(tuple(arr.shape))
        print(f"  • {out_meta.name:32s} → shape {shape_str:<20s} dtype {arr.dtype}")
