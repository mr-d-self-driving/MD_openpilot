#!/usr/bin/env python3
"""
Robust ONNX → PyTorch → ONNX round-trip tester (OpenPilot edition).

  pip install --upgrade torch onnx onnxruntime onnx2torch packaging
"""

import argparse, numpy as np, torch, onnx, onnxruntime as ort
from pathlib import Path
from typing import Dict, List
from onnx import helper, shape_inference, TensorProto

# --------------------------------------------------------------------------- #
# 1.  Random input generator                                                  #
# --------------------------------------------------------------------------- #
def _random_feed(model: onnx.ModelProto, batch: int, seed: int = 0
                 ) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feed = {}
    dtype_map = {k: helper.tensor_dtype_to_np_dtype(k) for k in range(1, 18)}
    for vi in model.graph.input:
        if any(vi.name == init.name for init in model.graph.initializer):
            continue                                   # skip constant weights
        shape = [(d.dim_value if d.dim_value > 0 else batch)
                 for d in vi.type.tensor_type.shape.dim]
        feed[vi.name] = rng.standard_normal(shape).astype(
            dtype_map[vi.type.tensor_type.elem_type])
    return feed

# --------------------------------------------------------------------------- #
# 2.  Monkey-patch converters that mishandle dynamic axes                     #
# --------------------------------------------------------------------------- #
import importlib
def _replace_op(mod_path: str, cls_name: str, new_cls):
    mod = importlib.import_module(mod_path)
    if hasattr(mod, cls_name):
        setattr(mod, cls_name, new_cls)

class SafeReshape(torch.nn.Module):
    def forward(self, x, shape):
        shape = [x.size(i) if s == 0 else int(s)
                 for i, s in enumerate(shape.to(torch.long).tolist())]
        return torch.reshape(x, tuple(shape))

class SafeSplit(torch.nn.Module):
    def forward(self, x, split, axis):                 # split is 1-D tensor
        return torch.split(x, split.tolist(), dim=int(axis))

class SafeSqueeze(torch.nn.Module):
    def forward(self, x, axes):
        return torch.squeeze(x,
                             dim=tuple(axes.tolist()) if axes.numel() else None)

class SafeClip(torch.nn.Module):                       # fp16-safe clamp
    def __init__(self, min=None, max=None):
        super().__init__(); self.min=min; self.max=max
    def forward(self, x): return torch.clamp(x, min=self.min, max=self.max)

class SafeSigmoid(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x).to(x.dtype)          # keeps original dtype

def _wrap_fp16(module: torch.nn.Module):
    orig_fwd = module.forward
    def new_fwd(x, *a, **kw):
        y = orig_fwd(x, *a, **kw)
        return y.to(x.dtype)        # round to same dtype as input
    module.forward = new_fwd


_replace_op("onnx2torch.node_converters.reshape",  "OnnxReshape",   SafeReshape)
_replace_op("onnx2torch.node_converters.split",    "OnnxSplit",     SafeSplit)
_replace_op("onnx2torch.node_converters.squeeze",  "OnnxSqueezeDynamicAxes",
            SafeSqueeze)
_replace_op("onnx2torch.node_converters.clip",     "OnnxClip",      SafeClip)

# --------------------------------------------------------------------------- #
# 3.  Optional bisector – finds 1st mismatching node                          #
# --------------------------------------------------------------------------- #
def first_mismatch(model: onnx.ModelProto, pt: torch.nn.Module,
                   feed: Dict[str,np.ndarray], tol: float = 1e-3,
                   dump_dir: str = "dbg") -> None:
    import os, numpy as np, shutil
    g = onnx.ModelProto(); g.CopyFrom(model)

    # build name → (dtype, shape) map
    vi_cache = {}
    def _add(name, dtype, dims):
        vi_cache[name] = (dtype, list(dims) if dims is not None else [])
    for coll in (g.graph.value_info, g.graph.input, g.graph.output):
        for vi in coll:
            tt = vi.type.tensor_type
            _add(vi.name, tt.elem_type, [d.dim_value for d in tt.shape.dim])
    for init in g.graph.initializer:
        _add(init.name, init.data_type, init.dims)

    # extend outputs with every intermediate tensor
    seen = {o.name for o in g.graph.output}

    for node in g.graph.node:
        # (A) ignore synthetic Identity nodes that just forward a single input
        if (node.op_type == "Identity"
            and node.input
            and node.input[0].endswith("_output_0")):        # heuristic
            continue

        for out_name in node.output:
            # (B) ignore synthetic aliases
            if (out_name in seen
                or out_name.endswith(".value")
                or out_name.startswith("Identity_")):
                continue

            seen.add(out_name)
            dtype, shape = vi_cache.get(out_name,
                                        (TensorProto.FLOAT, []))
            g.graph.output.append(
                helper.make_tensor_value_info(out_name,
                                              dtype,
                                              shape or None))
    tmp_path = "/tmp/_dbg_all_outputs.onnx"; onnx.save(g, tmp_path)

    sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    ort_outs  = sess.run(None, feed)
    ort_names = [o.name for o in sess.get_outputs()]
    ort_dict  = {f"{n}:{i}": v
                 for n, vlist in zip(ort_names, ort_outs)
                 for i, v in enumerate(vlist if isinstance(vlist, (list,tuple))
                                       else [vlist])}

    pt_dict = {}
    def _hook(name):
        def fn(_, __, out):
            outs = out if isinstance(out, (list,tuple)) else (out,)
            for i,t in enumerate(outs):
                pt_dict[f"{name}:{i}"] = t.detach().cpu().numpy()
        return fn
    hooks=[]
    for name,m in pt.named_modules(): hooks.append(m.register_forward_hook(_hook(name)))
    pt(*tuple(torch.from_numpy(feed[n]) for n in feed))
    for h in hooks: h.remove()

    Path(dump_dir).mkdir(exist_ok=True)
    for name in ort_names:
        if name not in pt_dict: continue
        diff = np.max(np.abs(ort_dict[name]-pt_dict[name]))
        if diff>tol:
            np.save(Path(dump_dir)/f"ort_{name}.npy", ort_dict[name])
            np.save(Path(dump_dir)/f"pt_{name}.npy",  pt_dict[name])
            print(f"❌ first mismatch {name}  |Δ|={diff:.4e}  tensors in {dump_dir}/")
            return
    print("✅ all intermediates within tol")

# --------------------------------------------------------------------------- #
# 4.  Main                                                                    #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/adas/openpilot/openpilot/selfdrive/modeld/models/driving_policy_with_normal_nav.onnx")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--out",   default="/home/adas/openpilot/openpilot/selfdrive/modeld/models/roundtrip.onnx")
    p.add_argument("--tol",   type=float, default=1e-3)
    p.add_argument("--bisect", action="store_true",
                   help="run first-mismatch debugger once then exit")
    args = p.parse_args()

    # load & infer shapes
    orig = shape_inference.infer_shapes(onnx.load(args.model))

    # deterministic random inputs
    feed = _random_feed(orig, batch=args.batch)

    # baseline ORT
    sess_orig = ort.InferenceSession(args.model,
                                     providers=["CPUExecutionProvider"])
    ort_ref   = sess_orig.run(None, feed)

    # ONNX → PyTorch
    from onnx2torch import convert
    torch_model = convert(orig).eval()
    with torch.no_grad():
        for p_ in torch_model.parameters(): p_.data = p_.data.clone()

    # bisect option
    if args.bisect:
        first_mismatch(orig, torch_model, feed, tol=args.tol)
        return

    for m in torch_model.modules():
        if m.__class__.__name__ in {"OnnxSigmoid", "OnnxHardSigmoid",
                                    "OnnxHardsigmoid", "OnnxLogSigmoid",
                                    "OnnxClip"}:           # Clip already safe but ok
            _wrap_fp16(m)
    # prepare inputs in declared order
    input_names = [vi.name for vi in orig.graph.input
                   if vi.name not in {i.name for i in orig.graph.initializer}]
    torch_inputs = tuple(torch.from_numpy(feed[n]) for n in input_names)

    # forward, then cast each output to ONNX dtype
    with torch.no_grad():
        outs = torch_model(*torch_inputs)
    outs = outs if isinstance(outs,(list,tuple)) else [outs]
    for i,(t,info) in enumerate(zip(outs, sess_orig.get_outputs())):
        if info.type == "tensor(float16)": outs[i]=t.to(torch.float16)

    outs_np = [t.detach().cpu().numpy() for t in outs]

    # export back to ONNX
    torch.onnx.export(
        torch.nn.ModuleList([torch_model]).eval()[0],   # simple wrapper
        torch_inputs,
        args.out,
        opset_version=17,
        input_names=input_names,
        output_names=[o.name for o in sess_orig.get_outputs()],
        dynamic_axes={n:{0:"batch"} for n in input_names}
    )

    # run exported
    ort_round = ort.InferenceSession(args.out,
                                     providers=["CPUExecutionProvider"]).run(None, feed)

    # compare by name
    ort_dict = {o.name:t for o,t in zip(sess_orig.get_outputs(), ort_ref)}
    pyt_dict = {o.name:t for o,t in zip(sess_orig.get_outputs(), outs_np)}

    def _err(a,b):
        if a.dtype!=b.dtype:
            a=a.astype(np.float32); b=b.astype(np.float32)
        return np.abs(a-b).max()

    max_orig  = max(_err(ort_dict[n], pyt_dict[n]) for n in ort_dict)
    max_round = max(_err(pyt_dict[n], ort_round[i])
                    for i,n in enumerate(ort_dict))

    for n in ort_dict:
        print(f"outputs {n} {ort_dict[n].dtype}{ort_dict[n].shape} "
              f"{pyt_dict[n].dtype}{pyt_dict[n].shape}")
    print(f"max |ONNX(orig) – PyTorch|  = {max_orig:.4e}")
    print(f"max |PyTorch – ONNX(exp)|  = {max_round:.4e}")
    if max(max_orig,max_round)<=args.tol:
        print("✅  Round-trip PASSED")
    else:
        print("❌  Round-trip FAILED")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
