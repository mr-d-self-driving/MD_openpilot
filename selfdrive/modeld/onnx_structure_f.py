#!/usr/bin/env python3
"""
inject_nav_inputs.py (v2)
========================
Patch two places inside an existing OpenPilot **driving_policy.onnx** so
that navigation information is fed into both the *plan* and *action* heads.

### New components
| # | Name          | Shape | Note |
|---|---------------|-------|------|
| 1 | **NavBranch** | (1,256) → (1,256) | two-block residual MLP, zero-init |
| 2 | **Concat-plan** | (…,512+406) | feeds `…/plan/Gemm` |
| 3 | **Concat-act**  | (…,N+406)   | feeds first Gemm in action head |

Both downstream Gemm weights are *right-padded* with 406 zeros; biases stay
unchanged.

After running, **driving_policy_with_nav.onnx** is ready for fine-tuning.

Dependencies
------------
```bash
pip install onnx==1.15 onnx-graphsurgeon numpy onnxruntime
```
"""
from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import onnx
import onnx_graphsurgeon as gs

# --------------------------------------------------------------------------- #
# 0 · Helper: give every new node a unique name                                #
# --------------------------------------------------------------------------- #
_uid = itertools.count().__next__  # global counter
def N(op: str, **kwargs):          # short for "named node"
    """Return a gs.Node with a unique, readable name if none provided."""
    kwargs.setdefault("name", f"{op}_{_uid()}")
    return gs.Node(op, **kwargs)

# --------------------------------------------------------------------------- #
# 1 · Load original model                                                      #
# --------------------------------------------------------------------------- #
ROOT = Path('/home/adas/openpilot/selfdrive/modeld/models')
IN_FILE  = ROOT / 'driving_policy.onnx'
OUT_FILE = ROOT / 'driving_policy_with_normal_nav.onnx'
print(f"Loading {IN_FILE} …")

model = onnx.load(str(IN_FILE))
graph = gs.import_onnx(model)

after = lambda *nodes: graph.nodes.extend([n for n in nodes if n not in graph.nodes])

# --------------------------------------------------------------------------- #
# 2 · Add graph inputs                                                         #
# --------------------------------------------------------------------------- #
nav_feat  = gs.Variable('nav_features',  dtype=np.float16, shape=(1,256))
nav_instr = gs.Variable('nav_instructions', dtype=np.float16, shape=(1,150))
for v in (nav_feat, nav_instr):
    if v.name not in {i.name for i in graph.inputs}:
        graph.inputs.append(v)

zeros = lambda name, shape: gs.Constant(name, values=np.zeros(shape, np.float16))

# --------------------------------------------------------------------------- #
# 3 · Build NavBranch (2 × residual + projection)                              #
# --------------------------------------------------------------------------- #

def residual(prefix: str, x: gs.Variable, width: int = 256, hidden: int = 512):
    fc1_out = gs.Variable(f"{prefix}_fc1_out", dtype=np.float16)
    fc1 = N('Gemm', inputs=[x, zeros(f"{prefix}_W1", (width, hidden)), zeros(f"{prefix}_b1", (hidden,))], outputs=[fc1_out])
    relu1 = N('Relu', inputs=[fc1_out], outputs=[gs.Variable(f"{prefix}_relu1", dtype=np.float16)])

    fc2_out = gs.Variable(f"{prefix}_fc2_out", dtype=np.float16)
    fc2 = N('Gemm', inputs=[relu1.outputs[0], zeros(f"{prefix}_W2", (hidden, width)), zeros(f"{prefix}_b2", (width,))], outputs=[fc2_out])

    add_out = gs.Variable(f"{prefix}_add", dtype=np.float16)
    add = N('Add', inputs=[fc2_out, x], outputs=[add_out])
    relu2 = N('Relu', inputs=[add_out], outputs=[gs.Variable(f"{prefix}_relu2", dtype=np.float16)])
    after(fc1, relu1, fc2, add, relu2)
    return relu2.outputs[0]

# blk1 = residual('nav_blk1', nav_feat)
# blk2 = residual('nav_blk2', blk1)

# nav_vec = gs.Variable('nav_vec', dtype=np.float16)
# after(N('Gemm', inputs=[blk2, zeros('nav_proj_W', (256,256)), zeros('nav_proj_b', (256,))], outputs=[nav_vec]))
# nav_vec_relu = gs.Variable('nav_vec_relu', dtype=np.float16)
# after(N('Relu', inputs=[nav_vec], outputs=[nav_vec_relu]))

# --------------------------------------------------------------------------- #
# 4 · Helper to pad Gemm weights                                               #
# --------------------------------------------------------------------------- #
PAD_COLS = 256 + 150

def pad_W(old: gs.Constant, name: str):
    rows, cols = old.values.shape
    pad = np.zeros((rows, PAD_COLS), dtype=old.values.dtype)
    return gs.Constant(name, values=np.concatenate([old.values, pad], axis=1))

# --------------------------------------------------------------------------- #
# 5 · Patch PLAN head                                                          #
# --------------------------------------------------------------------------- #
plan_gemm = next(n for n in graph.nodes if n.op == 'Gemm' and n.name.endswith('/plan/Gemm'))
plan_in   = plan_gemm.inputs[0]
plan_concat_out = gs.Variable('plan_concat_in', dtype=np.float16)
after(N('Concat', inputs=[plan_in, nav_feat, nav_instr], outputs=[plan_concat_out], attrs={'axis': 1}))
plan_gemm.inputs[0] = plan_concat_out
plan_gemm.inputs[1] = pad_W(plan_gemm.inputs[1], 'plan_weight_extended')

# --------------------------------------------------------------------------- #
# 6 · Patch ACTION head                                                        #
# --------------------------------------------------------------------------- #
action_gemm = next(n for n in graph.nodes if n.op == 'Gemm' and n.name.endswith('/action_block/action_block_in/action_block_in.0/Gemm'))
act_in      = action_gemm.inputs[0]
action_concat_out = gs.Variable('action_concat_in', dtype=np.float16)
after(N('Concat', inputs=[act_in, nav_feat, nav_instr], outputs=[action_concat_out], attrs={'axis': 1}))
action_gemm.inputs[0] = action_concat_out
action_gemm.inputs[1] = pad_W(action_gemm.inputs[1], 'action_weight_extended')

# --------------------------------------------------------------------------- #
# 7 · Fix Add broadcast order for onnx2pytorch                                 #
# --------------------------------------------------------------------------- #
for add in [n for n in graph.nodes if n.op == 'Add' and len(n.inputs) == 2]:
    a, b = add.inputs
    if isinstance(a, gs.Constant) and a.values.ndim == 1 and not isinstance(b, gs.Constant):
        add.inputs = [b, a]

# --------------------------------------------------------------------------- #
# 8 · Save                                                                     #
# --------------------------------------------------------------------------- #
graph.cleanup().toposort()
proto = gs.export_onnx(graph)
proto.ir_version = 10
onnx.save(proto, str(OUT_FILE))
print(f"Saved → {OUT_FILE}")

# --------------------------------------------------------------------------- #
# 9 · Optional sanity-check                                                    #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    try:
        import onnxruntime as ort
        print('Running dummy forward-pass …')
        sess = ort.InferenceSession(str(OUT_FILE), providers=['CPUExecutionProvider'])
        dummy = {
            inp.name: np.zeros([1 if s is None else s for s in inp.shape], np.float16 if inp.type == 'tensor(float16)' else np.float32)
            for inp in sess.get_inputs()
        }
        outs = sess.run(None, dummy)
        print('✓ OK —', len(outs), 'output tensors')
    except ImportError:
        print('onnxruntime not installed — skipped sanity-check.')