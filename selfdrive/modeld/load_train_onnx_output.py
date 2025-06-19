#!/usr/bin/env python3
"""
nav_finetune_fp32.py  (frame-by-frame, no extra batch-dim)
=========================================================

Fine-tune the **nav-aware** OpenPilot driving-policy model strictly in
**FP32 precision**.  All parameters, inputs, and targets are kept as
`torch.float32`; no mixed-precision or FP16 casts are performed.  Only the
`Gemm` layers downstream of `nav_features` / `nav_instructions` are
updated; the rest of the network is frozen.

Example
-------
```bash
python nav_finetune_fp32.py \
       --logdir /home/adas/openpilot/policy_logger \
       --onnx   /home/adas/openpilot/selfdrive/modeld/models/driving_policy_with_nav.onnx \
       --epochs 3 --lr 2e-4 --device cuda \
       --save   ./policy_finetuned_nav.pt
```
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any

import onnx
import torch
from onnx2pytorch import ConvertModel
import torch.nn.functional as F
from random import shuffle
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# 0 · CLI                                                                     #
# --------------------------------------------------------------------------- #
P = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
P.add_argument("--logdir", type=Path, default="/home/adas/openpilot/policy_logger",
               help="Folder with batch_*.pt shards from policy_logger")
P.add_argument("--onnx",  type=Path,
               default="/home/adas/openpilot/selfdrive/modeld/models/driving_policy_with_nav.onnx",
               help="Path to nav-aware ONNX graph")
P.add_argument("--epochs", type=int,   default=100,    help="Fine-tuning epochs")
P.add_argument("--lr",     type=float, default=2e-4, help="Learning rate")
P.add_argument("--device", choices=["cpu", "cuda"],
               default="cuda" if torch.cuda.is_available() else "cpu")
P.add_argument("--save",   type=Path,  default=None, help="Path to save state-dict")
P.add_argument("--save_onnx", type=Path, default='/home/adas/openpilot/selfdrive/modeld/models/driving_policy_with_nav_finetuned.onnx',
               help="If given, write an ONNX file with updated Gemm weights")
args = P.parse_args()

DEVICE = torch.device(args.device)
print(f"▶  Device: {DEVICE} | Epochs: {args.epochs} | Updates per epoch ≙ #frames")

# --------------------------------------------------------------------------- #
# 1 · Replay iterator                                                         #
# --------------------------------------------------------------------------- #
class PolicyReplay:
    """Iterate over `policy_logger` shards frame-by-frame (no batch dim added)."""

    def __init__(self, shard_dir: Path):
        self.files = sorted(shard_dir.glob("batch_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No batch_*.pt shards in {shard_dir}")
        self.index: List[Tuple[int, int]] = []
        for f_idx, f in enumerate(self.files):
            num = len(torch.load(f, map_location="cpu"))
            self.index.extend([(f_idx, i) for i in range(num)])
        print(f"✔  {len(self.index):,} frames across {len(self.files)} shards found")

    def __len__(self):
        return len(self.index)

    def get(self, i: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Return input dict and float32 target tensor on *CPU*."""
        f_idx, rec_idx = self.index[i]
        rec = torch.load(self.files[f_idx], map_location="cpu")[rec_idx]
        inp  = rec["in"]            # dict[str, np.ndarray | Tensor]
        tgt  = torch.as_tensor(rec["out"], dtype=torch.float32)  # (5884,)
        return inp, tgt

replay = PolicyReplay(args.logdir)

# --------------------------------------------------------------------------- #
# 2 · ONNX graph analysis                                                     #
# --------------------------------------------------------------------------- #
print(f"▶  Parsing ONNX graph: {args.onnx}")
onx = onnx.load(str(args.onnx))
consumers: Dict[str, List[onnx.NodeProto]] = defaultdict(list)
for node in onx.graph.node:
    for inp in node.input:
        consumers[inp].append(node)

nav_sources = ("nav_features", "nav_instructions")
init_names, gemm_nodes = set(), set()
queue, seen = deque(nav_sources), set()
while queue:
    tensor = queue.popleft()
    for node in consumers.get(tensor, []):
        if id(node) in seen:
            continue
        seen.add(id(node))
        if node.op_type == "Gemm":
            gemm_nodes.add(node.name)
            if len(node.input) >= 2:
                init_names.add(node.input[1])
            if len(node.input) == 3:
                init_names.add(node.input[2])
        queue.extend(node.output)
print(f"✔  {len(gemm_nodes)} downstream Gemm nodes → {len(init_names)} weight/bias tensors")

# --------------------------------------------------------------------------- #
# 3 · Convert to PyTorch & freeze                                             #
# --------------------------------------------------------------------------- #
print("▶  Converting ONNX → PyTorch …")
# ⚠️  IMPORTANT: force FP32 immediately after conversion
model = ConvertModel(onx).float().to(DEVICE).train()


def is_trainable(pname: str) -> bool:
    """Return True if `pname` belongs to a Gemm weight/bias that follows nav inputs."""
    base_names = {n.lstrip("/") for n in init_names}
    if any(tok in pname for tok in init_names | base_names):
        return True
    if any(f"Gemm_{n}" in pname for n in gemm_nodes):
        return True
    return ("nav_" in pname) or ("weight_extended" in pname)

for name, p in model.named_parameters():
    p.requires_grad_(is_trainable(name))
trainable_cnt = sum(p.requires_grad for p in model.parameters())
print(f"✔  {trainable_cnt} tensors marked trainable")
if trainable_cnt == 0:
    raise RuntimeError("No parameters marked trainable—check matching logic.")

# --------------------------------------------------------------------------- #
# 4 · Optimiser & loss                                                        #
# --------------------------------------------------------------------------- #
optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                         lr=args.lr, weight_decay=1e-4)
print(f"✔  AdamW initialised | lr={args.lr}")

# --------------------------------------------------------------------------- #
# 5 · Fine-tuning loop                                                        #
# --------------------------------------------------------------------------- #
loss_history: List[float] = []
for epoch in range(1, args.epochs + 1):
    idxs = list(range(len(replay)))
    shuffle(idxs)
    running_loss = 0.0

    pbar = tqdm(idxs, desc=f"Epoch {epoch}/{args.epochs}")
    for i in pbar:
        inp_raw, tgt_cpu = replay.get(i)

        # -- prepare tensors on DEVICE, all as FP32 --
        inputs: Dict[str, torch.Tensor] = {}
        for k, v in inp_raw.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device=DEVICE, dtype=torch.float32)
            else:  # numpy array or list
                inputs[k] = torch.as_tensor(v, device=DEVICE, dtype=torch.float32)
        target = tgt_cpu.to(device=DEVICE, dtype=torch.float32)

        optim.zero_grad(set_to_none=True)
        preds = model(**inputs)
        # some ConvertModel outputs a (1, N) tensor — squeeze the batch dim if so
        preds = preds.squeeze(0) if preds.dim() == 2 and preds.size(0) == 1 else preds

        loss  = F.mse_loss(preds, target)
        loss.backward()
        optim.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg = running_loss / len(replay)
    loss_history.append(avg)
    print(f"Epoch {epoch}/{args.epochs} – avg MSE: {avg:.6f}")

print("✓ Fine-tuning complete")

# --------------------------------------------------------------------------- #
# 6 · Save                                                                    #
# --------------------------------------------------------------------------- #
if args.save:
    torch.save({"state_dict": model.state_dict(),
                "loss_history": loss_history,
                "trainable_param_count": trainable_cnt},
               args.save)
    print("✓ State-dict saved to", args.save)

if args.save_onnx:
    # 1.  Get the canonical input order from the original graph
    in_names = [i.name for i in onx.graph.input]          # preserves order
    print(in_names)
    # 2.  Build a matching tuple of dummy tensors
    sample_inp, _ = replay.get(0)
    args_tuple = tuple(
        torch.as_tensor(sample_inp[n], dtype=torch.float32, device="cpu")
        for n in in_names
    )

    # 3.  Thin wrapper that forwards *positional* args
    class Wrapper(torch.nn.Module):
        def __init__(self, mdl): super().__init__(); self.m = mdl.cpu().eval()
        def forward(self, *inputs): return self.m(*inputs)

    out_path = Path(args.save_onnx).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        Wrapper(model),               # model
        args_tuple,                   # positional example inputs
        f=out_path.as_posix(),        # file
        input_names=in_names,
        output_names=["out"],
        dynamic_axes={n: {0: "N"} for n in in_names} | {"out": {0: "N"}},
        opset_version=17,
    )
    print("✓ Finetuned model exported as ONNX →", out_path)