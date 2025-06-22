#!/usr/bin/env python3
"""
onnx_train.py
=============
Load **driving_policy_with_nav.onnx**, convert it to PyTorch, and leave *only*
Gemm layers that lie downstream of `nav_features`/`nav_instructions` (plus the
NavBranch and padded head matrices) trainable. Expected: **38** trainable
weight/bias tensors out of ~113 total.

Run:
    python onnx_train.py  # prints summary, no training loop yet

Add your own DataLoader, loss function, and optimisation loop where indicated.
"""
from pathlib import Path
from collections import defaultdict, deque
import onnx
import torch
from onnx2pytorch import ConvertModel

# ---------------------------------------------------------------------------
# 0 · Paths & Device
# ---------------------------------------------------------------------------
ROOT       = Path('/home/adas/openpilot/selfdrive/modeld/models')
ONNX_FILE  = ROOT / 'driving_policy_with_nav.onnx'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# 1 · Load and Parse ONNX Graph
# ---------------------------------------------------------------------------
print(f"Loading {ONNX_FILE} …")
onnx_model = onnx.load(str(ONNX_FILE))

# Build tensor→consumer lookup
consumers = defaultdict(list)
for node in onnx_model.graph.node:
    for inp in node.input:
        consumers[inp].append(node)

# BFS: collect Gemm nodes and their weight/bias initializer names downstream of nav_
nav_sources = ['nav_features', 'nav_instructions']
init_names, gemm_nodes = set(), set()
queue, seen = deque(nav_sources), set()
while queue:
    tensor = queue.popleft()
    for node in consumers.get(tensor, []):
        if id(node) in seen:
            continue
        seen.add(id(node))
        if node.op_type == 'Gemm':
            gemm_nodes.add(node.name)
            # weight is second input, bias optional third
            if len(node.input) >= 2:
                init_names.add(node.input[1])
            if len(node.input) == 3:
                init_names.add(node.input[2])
        queue.extend(node.output)

print(f"{len(gemm_nodes)} Gemm nodes downstream of nav_* → {len(init_names)} weight/bias tensors")

# ---------------------------------------------------------------------------
# 2 · Convert ONNX to PyTorch
# ---------------------------------------------------------------------------
print("Converting ONNX → PyTorch …")
policy_pt = ConvertModel(onnx_model).to(DEVICE).train()

# ---------------------------------------------------------------------------
# 3 · Define Matching Logic for Trainable Params
# ---------------------------------------------------------------------------
def is_nav_gemm_param(pname: str) -> bool:
    """Return True if pname corresponds to a downstream Gemm weight/bias."""
    # ① match on initializer names as before
    for init in init_names:
        base = init.lstrip('/')
        variants = [
            init,                 # exact initializer
            base,                 # without slash
            '/' + base,           # with slash
            f"Gemm_{base}",      # onnx2pytorch prefix of initializer
            f"Gemm_/{base}",     # with slash prefix
        ]
        if any(v in pname for v in variants):
            return True
    # ② match on Gemm node names directly (handles downstream heads)
    for node_name in gemm_nodes:
        if f"Gemm_{node_name}" in pname:
            return True
    # always keep nav branch and padded head weights
    if 'nav_' in pname or 'weight_extended' in pname:
        return True
    return False

# ---------------------------------------------------------------------------
# 4 · Freeze/Unfreeze and Report
# ---------------------------------------------------------------------------
trainable, frozen = [], []
for name, param in policy_pt.named_parameters():
    keep = is_nav_gemm_param(name)
    param.requires_grad_(keep)
    (trainable if keep else frozen).append(name)

# Report counts
print(f"{len(trainable)} trainable / {len(trainable)+len(frozen)} total params")
print(f"{len(frozen)} frozen parameters")
for n in trainable:
    print(f"  • {n}")

assert trainable, "No trainable parameters found—check matching logic!"

# ---------------------------------------------------------------------------
# 5 · Build Optimizer
# ---------------------------------------------------------------------------
optim = torch.optim.AdamW(
    (p for p in policy_pt.parameters() if p.requires_grad),
    lr=2e-4,
    weight_decay=1e-4
)
print(f"\n✓ AdamW initialised with {len(optim.param_groups[0]['params'])} tensors")

# ---------------------------------------------------------------------------
# 6 · Training Loop Placeholder
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("\nAdd your DataLoader, loss_fn, and training loop below this line.")
