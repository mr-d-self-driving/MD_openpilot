from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# optional ONNX import (only needed for --onnx / --save)
try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    onnx = None


# --------------------------------------------------------------------------- #
#                               MODEL
# --------------------------------------------------------------------------- #
class TemporalSummarizer(nn.Module):
    DESIRE_SHAPE = (25, 8)                        # (time, features)
    DESIRE_DIM   = DESIRE_SHAPE[0] * DESIRE_SHAPE[1]   # 25 × 8 = 200
    TC_DIM       = 2                              # traffic_convention size
    HIDDEN       = 512
    GATHER_IDX   = 9                              # Gather(index = 9)

    def __init__(self) -> None:
        super().__init__()

        # /temporal_summarizer/extra_in.0/Gemm
        self.temporal_summarizer = nn.Module()
        self.temporal_summarizer.extra_in = nn.ModuleList([
            nn.Linear(self.DESIRE_DIM + self.TC_DIM, self.HIDDEN)
        ])

        # /temporal_summarizer/encode.0/MatMul (+ Add)
        self.temporal_summarizer.encode = nn.ModuleList([
            nn.Linear(self.HIDDEN, self.HIDDEN)
        ])

    # --------------------------------------------------------------------- #
    def forward(
        self,
        traffic_convention: torch.Tensor,  # [B, 2]
        desire: torch.Tensor,             # [B, 25, 8]
        features_buffer: torch.Tensor     # [B, 25, 512]
    ) -> torch.Tensor:
        # --- Reshape + Concat ------------------------------------------- #
        B = desire.shape[0]
        desire_flat = desire.view(B, -1)                         # [B, 200]
        extra_in    = torch.cat([traffic_convention, desire_flat], dim=-1)  # [B, 202]

        # Gemm → ReLU
        extra = F.relu(self.temporal_summarizer.extra_in[0](extra_in))      # [B, 512]
        extra = extra.unsqueeze(1)                                          # [B, 1, 512]

        # Gather idx 9 from features_buffer
        gathered = features_buffer[:, self.GATHER_IDX:self.GATHER_IDX+1]    # [B, 1, 512]

        # Concat → Linear → ReLU
        seq     = torch.cat([gathered, extra], dim=1)                       # [B, 2, 512]
        encoded = F.relu(self.temporal_summarizer.encode[0](seq))           # [B, 2, 512]
        return encoded


# --------------------------------------------------------------------------- #
#                OPTIONAL :  LOAD  +  SAVE  ONNX
# --------------------------------------------------------------------------- #
def load_onnx_weights(model: TemporalSummarizer, onnx_path: Path) -> None:
    if onnx is None:
        raise RuntimeError("onnx not installed; pip install onnx")

    g = onnx.load(str(onnx_path)).graph
    wt = {t.name: numpy_helper.to_array(t) for t in g.initializer}

    model.temporal_summarizer.extra_in[0].weight.data.copy_(
        torch.from_numpy(wt["/temporal_summarizer/extra_in.0/Gemm/B"])
    )
    model.temporal_summarizer.extra_in[0].bias.data.copy_(
        torch.from_numpy(wt["/temporal_summarizer/extra_in.0/Gemm/C"])
    )
    model.temporal_summarizer.encode[0].weight.data.copy_(
        torch.from_numpy(wt["/temporal_summarizer/encode.0/MatMul/B"])
    )
    model.temporal_summarizer.encode[0].bias.data.copy_(
        torch.from_numpy(wt["/temporal_summarizer/encode.0/Add/A"])
    )
    print("✓  ONNX weights imported")


def export_to_onnx(
    model: TemporalSummarizer,
    save_path: Path,
    batch: int = 1,
    time: int = 25,          # must be ≥ 25 so Gather(idx 9) is valid
) -> None:
    if onnx is None:
        raise RuntimeError("onnx not installed; pip install onnx")

    model.eval()

    tc   = torch.zeros(batch, 2)                   # dummy traffic_convention
    des  = torch.zeros(batch, *model.DESIRE_SHAPE) # dummy desire
    buf  = torch.zeros(batch, time, 512)           # dummy features_buffer

    torch.onnx.export(
        model,
        (tc, des, buf),
        str(save_path),
        input_names=["traffic_convention", "desire", "features_buffer"],
        output_names=["encoded"],
        dynamic_axes={
            "traffic_convention": {0: "batch"},
            "desire":            {0: "batch"},
            "features_buffer":   {0: "batch", 1: "time"},
            "encoded":           {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"✓  exported → {save_path.resolve()}")

# --------------------------------------------------------------------------- #
#                                   CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path,default='/home/adas/openpilot/selfdrive/modeld/models/driving_policy.onnx', help="load weights from existing .onnx")
    ap.add_argument("--save", type=Path,default='policy_nav_only_fixedC.onnx', help="export PyTorch model to this .onnx")
    ap.add_argument("--batch", type=int, default=1, help="dummy batch size for export")
    ap.add_argument("--time",  type=int, default=12, help="dummy time length for export (≥10)")

    args = ap.parse_args()

    mdl = TemporalSummarizer()
    # if args.onnx:
    #     load_onnx_weights(mdl, args.onnx)

    # quick forward sanity check
    out = mdl(
        torch.randn(args.batch, 1),
        torch.randn(args.batch, 201),
        torch.randn(args.batch, args.time, 512),
    )
    print("forward OK, output shape:", tuple(out.shape))

    if args.save:
        export_to_onnx(mdl, args.save, batch=args.batch, time=args.time)


if __name__ == "__main__":
    main()
