import argparse, onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def is_scalar_initializer(g, name):
    for init in g.initializer:
        if init.name == name:
            return (init.dims == []
                    or (len(init.dims) == 1 and init.dims[0] == 1))
    return False

def clone_tensor(g, name):
    return next(i for i in g.initializer if i.name == name)

# --------------------------------------------------------------------------- #
def keep_scalar_identities(model):
    g = model.graph
    kept = []
    for n in g.node:
        if n.op_type == "Identity":
            if is_scalar_initializer(g, n.input[0]):
                kept.append(n)      # keep scalar forwarder
            # else: drop non-scalar Identity
        else:
            kept.append(n)
    g.ClearField("node")
    g.node.extend(kept)
    return model

# --------------------------------------------------------------------------- #
def replace_scalar_identities_with_constants(model):
    g = model.graph
    new_nodes = []
    for n in g.node:
        if n.op_type == "Identity":
            if is_scalar_initializer(g, n.input[0]):
                # redirect every consumer of its output to a fresh Constant
                out_name = n.output[0]
                weight   = clone_tensor(g, n.input[0])
                for m in g.node:
                    m.input[:] = [out_name if x == out_name else x for x in m.input]
                const_node = helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[out_name],
                    name=f"Const_{out_name}",
                    value=weight)
                new_nodes.append(const_node)
            # skip Identity itself
        else:
            new_nodes.append(n)
    g.ClearField("node")
    g.node.extend(new_nodes)
    return model

# --------------------------------------------------------------------------- #
def drop_all_identities(model):
    g = model.graph
    g.node[:] = [n for n in g.node if n.op_type != "Identity"]
    return model

# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ino",  default="/home/adas/openpilot/openpilot/selfdrive/modeld/models/roundtrip.onnx", help="input .onnx path")
    p.add_argument("--out", default="/home/adas/openpilot/openpilot/selfdrive/modeld/models/roundtrip1.onnx", help="output .onnx path")
    p.add_argument("--mode", choices=["identity", "constant", "drop"],
                   default="identity",
                   help="choose how to expose scalar constants")
    args = p.parse_args()

    model = onnx.load(args.ino)

    if args.mode == "identity":
        model = keep_scalar_identities(model)
    elif args.mode == "constant":
        model = replace_scalar_identities_with_constants(model)
    elif args.mode == "drop":
        model = drop_all_identities(model)

    onnx.save(model, args.out)
    print(f"✔  wrote '{args.out}' using mode='{args.mode}'")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()