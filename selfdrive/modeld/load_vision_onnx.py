import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import onnxruntime as rt
import onnx
import onnx2pytorch
# import onnx2keras

path_to_onnx_model = '/home/adas/openpilot/selfdrive/modeld/models/driving_vision_clean.onnx'

model = onnx.load(path_to_onnx_model)

input_names = [node.name for node in model.graph.input]
output_names = [node.name for node in model.graph.output]

# onnxruntime
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
onnxruntime_model = rt.InferenceSession(path_to_onnx_model, providers=providers)

# pytorch
device = torch.device('cuda')
pytorch_model = onnx2pytorch.ConvertModel(model).to(device)
pytorch_model.requires_grad_(False)
pytorch_model.eval()

# keras
# keras_model = onnx2keras.onnx_to_keras(model, input_names, verbose=False)

torch_inputs = {
    'input_imgs': torch.ones((1, 12, 128, 256), dtype=torch.uint8).to(device),
    'big_input_imgs': torch.ones((1, 12, 128, 256), dtype=torch.uint8).to(device),
}

onnx_inputs = {
    'input_imgs': np.ones((1, 12, 128, 256), dtype=np.uint8),
    'big_input_imgs': np.ones((1, 12, 128, 256), dtype=np.uint8),
}



# verify inputs are identical
for key in torch_inputs.keys():

  torch_val = torch_inputs[key].detach().cpu().numpy()
  onnx_val = onnx_inputs[key]

  np.testing.assert_equal(torch_val, onnx_val)


onnxruntime_outs = onnxruntime_model.run(output_names, onnx_inputs)[0]

torch_outs = pytorch_model(**torch_inputs)
torch_outs = torch_outs.detach().cpu().numpy()
print('Torch outs:', torch_outs.shape)

print('onnxruntime outs:', onnxruntime_outs.shape)


# run inference
# keras_outs = keras_model(keras_inputs)
torch_outs = pytorch_model(**torch_inputs)
torch_outs = torch_outs.detach().cpu().numpy()

onnxruntime_outs = onnxruntime_model.run(output_names, onnx_inputs)[0]

torch_onnx_diff = np.sum(np.abs(torch_outs - onnxruntime_outs))
# onnx_keras_diff = np.sum(np.abs(onnxruntime_outs - keras_outs))

# print diffs
# print(f'Torch vs Keras: {torch_keras_diff:.3e}')
print(f'Torch vs ONNX: {torch_onnx_diff:.3e}')

ckpt_path = "/home/adas/openpilot/selfdrive/modeld/models/driving_vision_state_dict.pt"
torch.save(pytorch_model.state_dict(), ckpt_path)