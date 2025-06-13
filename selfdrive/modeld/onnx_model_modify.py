import onnx, onnx_graphsurgeon as gs, numpy as np

path_in  = "/home/adas/openpilot/selfdrive/modeld/models/driving_policy.onnx"
path_out = "/home/adas/openpilot/selfdrive/modeld/models/driving_policy_aug.onnx"

graph = gs.import_onnx(onnx.load(path_in))

# 1) locate the Concat node you showed in the dump
concat = next(n for n in graph.nodes if n.name == "/Concat_1")

# 2) make a NEW Variable (float16 to match the rest of the net)
nav_features   = "nav_features"
nav_features_shape  = [1, 256]          # (batch=1, axis-1 length=16) -- adjust as needed
nav_features_input  = gs.Variable(name=nav_features, dtype=np.float16, shape=nav_features_shape)

# 3) register it as a graph input *and* feed it to Concat
graph.inputs.append(nav_features_input)
concat.inputs.append(nav_features_input)            # order doesn’t matter for Concat
           # order doesn’t matter for Concat

# 4) because Concat's axis = 1, pad the *second* dimension of the next weight
#    (assumes the very next node is a Gemm / MatMul with weights in slot [1])
next_node = concat.outputs[0].outputs[0]    # -> Gemm/MatMul
W = next_node.inputs[1]                     # weight initializer
W.values = np.concatenate([W.values,
                           np.zeros((W.values.shape[0], nav_features_shape[1]),
                                    dtype=W.values.dtype)], axis=1)


nav_instructions   = "nav_instructions"
nav_instructions_shape  = [1, 150]          # (batch=1, axis-1 length=16) -- adjust as needed
nav_instructions_input  = gs.Variable(name=nav_instructions, dtype=np.float16, shape=nav_instructions_shape)

# 3) register it as a graph input *and* feed it to Concat
graph.inputs.append(nav_instructions_input)
concat.inputs.append(nav_instructions_input)

next_node = concat.outputs[0].outputs[0]    # -> Gemm/MatMul
W = next_node.inputs[1]                     # weight initializer
W.values = np.concatenate([W.values,
                           np.zeros((W.values.shape[0], nav_instructions_shape[1]),
                                    dtype=W.values.dtype)], axis=1)

# 5) save
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), path_out)
print("✅  Model saved to", path_out)
