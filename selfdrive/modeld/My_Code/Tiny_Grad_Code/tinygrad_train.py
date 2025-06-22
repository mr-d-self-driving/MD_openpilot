#!/usr/bin/env python3
"""
tinygrad_train.py  (OpenPilot nav-fine-tune edition)
===================================================
Fine-tunes only the Gemm layers that are downstream of the two navigation
inputs — `nav_features` and `nav_instructions` — while keeping every other
weight frozen.  Works with the *vendored* tinygrad inside OpenPilot
(older API: OnnxRunner(model)  – no `device=` kwarg).

Usage
-----
# GPU kernels
GPU=1 FLOAT16=1 python tinygrad_train.py \
        --onnx   ~/openpilot/selfdrive/modeld/models/driving_policy_with_nav.onnx \
        --logdir ~/openpilot/policy_logger \
        --epochs 3 --lr 2e-4 --device gpu

# CPU fallback
python tinygrad_train.py --device cpu ...
"""
from __future__ import annotations

import argparse, os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import onnx
import numpy as np
import torch                                # *only* for loading shard files

from tinygrad import Tensor, dtypes
from extra.onnx import OnnxRunner
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save, get_state_dict
from tqdm import tqdm
from tinygrad.nn.state import safe_load, load_state_dict

# ---------------------------------------------------------------------------#
# ---------------------------------------------------------------------------
# Dataset loader -------------------------------------------------------------
# ---------------------------------------------------------------------------

class PolicyReplay:
    """Iterate over policy_logger shards (frame-by-frame, no batch dim)."""

    def __init__(self, shard_dir: Path):
        self.files = sorted(shard_dir.glob("batch_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No batch_*.pt shards in {shard_dir}")
        self.index: List[Tuple[int, int]] = []
        for f_idx, f in enumerate(self.files):
            num = len(torch.load(f, map_location="cpu"))
            self.index.extend((f_idx, i) for i in range(num))
        print(f"✔  {len(self.index):,} frames across {len(self.files)} shards")

    def __len__(self): return len(self.index)

    def get(self, i: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
      f_idx, rec_idx = self.index[i]
      rec = torch.load(self.files[f_idx], map_location="cpu")[rec_idx]

      # --- inputs ---
      inp = {k: (v.numpy() if torch.is_tensor(v) else v)
            for k, v in rec["in"].items()}

      # --- target ---
      out = rec["out"]
      if torch.is_tensor(out):
          tgt = out.numpy().astype(np.float32)
      else:
          tgt = out.astype(np.float32)          # already NumPy
      return inp, tgt


# ---------------------------------------------------------------------------
# Graph analysis: mark trainable tensors ------------------------------------
# ---------------------------------------------------------------------------

def mark_downstream_trainables(runner: OnnxRunner,
                               model_proto: onnx.ModelProto,
                               src_inputs=("nav_features","nav_instructions")
                              ) -> List[Tensor]:
    """
    Enable grad on every parameter that influences either nav_features
    or nav_instructions.  Uses model_proto.graph for traversal.
    """
    g             = model_proto.graph
    reach, q      = set(src_inputs), list(src_inputs)
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in g.node:
        for t in node.input:
            consumers.setdefault(t, []).append(node)

    while q:                                         # BFS
        t = q.pop(0)
        for node in consumers.get(t, []):
            if any(i in reach for i in node.input):
                for o in node.output:
                    if o not in reach:
                        reach.add(o); q.append(o)

    init_names = {init.name for init in g.initializer}
    train_set  = {i for node in g.node if any(j in reach for j in node.input)
                  for i in node.input if i in init_names}

    trainables = []
    for name, val in runner.graph_values.items():
        if isinstance(val, Tensor):
            val.requires_grad = name in train_set
            if val.requires_grad:
                trainables.append(val)

    total_tensors = sum(isinstance(v, Tensor) for v in runner.graph_values.values())
    print(f"✓  will fine-tune {len(trainables)} / {total_tensors} tensors "
          "(downstream of nav_*)")
    return trainables


# ---------------------------------------------------------------------------
# Loss ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) ** 2).mean()


# ---------------------------------------------------------------------------
# Training loop -------------------------------------------------------------
# ---------------------------------------------------------------------------

def train(args):
    # honour --device
    device_name = args.device.upper()
    if device_name == "GPU" and os.environ.get("GPU") != "1":
        os.environ["GPU"] = "1"         # enable GPU kernels if user forgot
    Tensor.training = True
    # 1. load ONNX once
    model_proto = onnx.load(args.onnx)

    # 2. wrap runner (old API)
    runner = OnnxRunner(model_proto)

    # 3. pick trainables
    trainables = mark_downstream_trainables(runner, model_proto)

    # 4. optimiser
    # opt = AdamW(trainables, lr=args.lr, betas=(0.9,0.999), weight_decay=1e-2)
    opt = AdamW(trainables, lr=args.lr, b1=0.9, b2=0.999, weight_decay=1e-2)

    # 5. dataset
    replay = PolicyReplay(Path(args.logdir))
    steps  = len(replay)
    OUTPUT_KEY = "outputs"       # adjust if you need another head

    # 6. loop
    print(device_name)
    start_epoch = 0
    if args.resume:
        load_checkpoint(runner=runner,ckpt_path=args.resume)

    # with Tensor.train():
    for epoch in range(start_epoch, args.epochs+1):
        avg = 0.0
        pbar = tqdm(range(steps), desc=f"epoch {epoch}/{args.epochs}")
        for i in pbar:
            inp_np, tgt_np = replay.get(i)

            feed = {k: Tensor(v.astype(np.float16), dtype=dtypes.float16,
                              device=device_name)
                    for k, v in inp_np.items()}

            pred = runner(feed)[OUTPUT_KEY]
            tgt  = Tensor(tgt_np.astype(np.float16), dtype=dtypes.float16,
                          device=device_name)
            loss = mse_loss(pred, tgt)

            opt.zero_grad(); loss.backward(); opt.step()

            avg += loss.numpy()
            if (i+1) % 200 == 0:
                pbar.set_postfix(loss=avg/200); avg = 0.0

        ckpt = f"nav_finetune_epoch{epoch}.safetensors"
        safe_save(get_state_dict(runner), ckpt)
        print(f"📝  saved {ckpt}")
    stem   = Path(args.onnx).with_suffix('').name             # “model”
    out_path = Path(args.onnx).with_parent(Path(args.onnx).parent)\
            .with_name(f"{stem}_ty_finetuned.onnx")
    dump_weights_back_into_onnx(runner, model_proto, str(out_path))
    print("🎉 finished fine-tuning")
    Tensor.training = False


# ---------------------------------------------------------------------------#
#  CLI ----------------------------------------------------------------------#
# ---------------------------------------------------------------------------#
def dump_weights_back_into_onnx(runner: OnnxRunner,
                                model_proto: onnx.ModelProto,
                                path_out: str):
    name_to_tensor = {k: v.numpy() for k, v in runner.graph_values.items()
                      if isinstance(v, Tensor)}
    for init in model_proto.graph.initializer:
        if init.name in name_to_tensor:
            arr = name_to_tensor[init.name]
            # ONNX stores FP16 as dtype 10, FP32 as 1, etc.
            init.raw_data = arr.tobytes()
            if arr.dtype == np.float16: init.data_type = 10
            elif arr.dtype == np.float32: init.data_type = 1
            elif arr.dtype == np.float64: init.data_type = 11
            else:                         init.data_type = 0  # fallback
    onnx.save(model_proto, path_out)
    print(f"💾  wrote fine-tuned ONNX → {path_out}")
from safetensors.numpy import load_file
import re

def load_checkpoint(runner, ckpt_path):
    from tinygrad.nn.state import safe_load, load_state_dict
    load_state_dict(runner, safe_load(ckpt_path))
    return 0

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--resume", type=str, default=None,
               help="path to *.safetensors checkpoint to keep training from")
    p.add_argument("--onnx",   default="/home/adas/openpilot/openpilot/selfdrive/modeld/models/driving_policy_with_normal_nav.onnx")
    p.add_argument("--logdir", default="/home/adas/openpilot/policy_logger")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr",     type=float, default=2e-4)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    return p.parse_args()

if __name__ == "__main__":
    args = parse()
    train(args)
