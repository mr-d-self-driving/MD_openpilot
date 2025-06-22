#!/usr/bin/env python3
import onnx
from pathlib import Path

def list_nodes(path):
    model = onnx.load(path)
    print(f"\n=== {Path(path).name} ===")
    for i, node in enumerate(model.graph.node):
        op   = node.op_type
        name = node.name if node.name else "(no name)"
        out  = node.output[0] if node.output else "(no output)"
        print(f"{i:4d} │ {op:<10} │ {name} │ → {out}")

list_nodes('/home/adas/openpilot/openpilot/selfdrive/modeld/models/driving_policy_with_nav.onnx')
list_nodes('/home/adas/openpilot/openpilot/selfdrive/modeld/models/driving_policy_with_nav_finetuned.onnx')
