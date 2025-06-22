import onnx
model = onnx.load("/home/adas/openpilot/selfdrive/modeld/models/driving_vision.onnx")
for node in model.graph.node:
    if node.op_type == "Clip":
        # Remove trailing "" placeholders
        node.input[:] = [i for i in node.input if i]
onnx.save(model, "/home/adas/openpilot/selfdrive/modeld/models/driving_vision_clean.onnx")
