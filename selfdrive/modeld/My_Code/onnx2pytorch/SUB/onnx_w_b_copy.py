#!/usr/bin/env python3
"""
replace_gemm_params.py
======================

Usage
-----
python replace_gemm_params.py \
       --target  model_with_nav.onnx \
       --donor   baseline_policy.onnx \
       --save    patched.onnx
"""
import argparse
import onnx
import numpy as np
from onnx import numpy_helper

def get_node(graph, name: str):
    for n in graph.node:
        if n.name == name:
            return n
    raise ValueError(f"Node '{name}' not found")

def get_initializer_tensor(model, tensor_name: str):
    for t in model.graph.initializer:
        if t.name == tensor_name:
            return t
    raise ValueError(f"Initializer '{tensor_name}' not found")

def replace_gemm_weights(target_model, donor_model,
                         target_node_name: str,
                         donor_node_name: str):
    tgt_node   = get_node(target_model.graph, target_node_name)
    donor_node = get_node(donor_model.graph, donor_node_name)

    # Gemm inputs: [A, B (weights), C (bias)]
    tgt_W_name, tgt_b_name       = tgt_node.input[1:3]
    donor_W_name, donor_b_name   = donor_node.input[1:3]

    tgt_W = get_initializer_tensor(target_model, tgt_W_name)
    tgt_b = get_initializer_tensor(target_model, tgt_b_name)
    donor_W = get_initializer_tensor(donor_model, donor_W_name)
    donor_b = get_initializer_tensor(donor_model, donor_b_name)

    # Sanity-check shapes
    if numpy_helper.to_array(tgt_W).shape != numpy_helper.to_array(donor_W).shape:
        raise RuntimeError("Weight shapes mismatch")
    if numpy_helper.to_array(tgt_b).shape != numpy_helper.to_array(donor_b).shape:
        raise RuntimeError("Bias shapes mismatch")

    # Copy raw_data (fast, avoids realloc)
    tgt_W.raw_data = donor_W.raw_data
    tgt_b.raw_data = donor_b.raw_data
    print(f"✓ Replaced weights/bias of '{target_node_name}' "
          f"with values from '{donor_node_name}'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default='/home/adas/openpilot/openpilot/selfdrive/modeld/models/driving_policy_with_nav.onnx',
                    help="Path to ONNX model whose Gemm you want to patch")
    ap.add_argument("--donor",  default='/home/adas/openpilot/openpilot/selfdrive/modeld/models/driving_policy_with_nav_finetuned.onnx',
                    help="ONNX model that provides the new parameters")
    ap.add_argument("--save",   default='/home/adas/openpilot/openpilot/selfdrive/modeld/models/driving_policy_with_nav_finetuned_clear.onnx',
                    help="Output filename for patched model")
    args = ap.parse_args()

    target_model = onnx.load(args.target)
    donor_model  = onnx.load(args.donor)

    replace_gemm_weights(target_model, donor_model,
                         target_node_name = "/action_block/action_block_out/Gemm",
                         donor_node_name  = "/m/Gemm_/action_block/action_block_out/Gemm_output_0/Gemm")

    # Optionally re-run shape-inference to refresh value-info
    target_model = onnx.shape_inference.infer_shapes(target_model)
    onnx.checker.check_model(target_model)   # consistency check
    onnx.save(target_model, args.save)
    print(f"✓ Patched model saved → {args.save}")

if __name__ == "__main__":
    main()
