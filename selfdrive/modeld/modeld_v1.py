from typing import List, Optional
#!/usr/bin/env python3
import os
from openpilot.system.hardware import TICI
import onnxruntime as ort
import numpy as np
USBGPU = "USBGPU" in os.environ
# if USBGPU:
#   os.environ['AMD'] = '1'
# elif TICI:
#   from openpilot.selfdrive.modeld.runners.tinygrad_helpers import qcom_tensor_from_opencl_address
#   os.environ['QCOM'] = '1'
# else:
#   os.environ['LLVM'] = '1'
#   os.environ['JIT'] = '2'
from tinygrad_repo.tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import torch
import time
import pickle
import numpy as np
import cereal.messaging as messaging
from cereal import car, log
from pathlib import Path
import pyopencl as cl
# from common.gpu.gpu import queue
from setproctitle import setproctitle
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.system import sentry
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.models.commonmodel_pyx import DrivingModelFrame, CLContext
import cv2
from numpy.polynomial.polynomial import polyval
import onnx2pytorch
import onnx
import csv
import sys
import argparse
import matplotlib.pyplot as plt



ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--route",  type=Path, default="/home/adas/openpilot/kalman_track_2025-06-12_10-50-29.csv",
                help="reference path (.csv or .rlog.bz2)")
ap.add_argument("--thresh", type=float, default=20,  help="match threshold (m)")
ap.add_argument("--n",      type=int,   default=100,  help="# future points to plot")
ap.add_argument("--hz",     type=float, default=100, help="UI refresh rate (Hz)")
ap.add_argument("--window", type=float, default=50,  help="half window size (m)")
args = ap.parse_args()

# ───────── geo helpers ─────────────────────────────────────────────────────
R = 6_378_137.0
deg2rad = np.deg2rad
def haversine(lat1, lon1, lat2, lon2):
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(deg2rad(lat1))*np.cos(deg2rad(lat2))*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def equirect(lat, lon, lat0, lon0):
    dlat = deg2rad(lat - lat0)
    dlon = deg2rad(lon - lon0)
    x = R*np.cos(deg2rad(lat0))*dlon
    y = R*dlat
    return x, y

def rotate(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return  x*c + y*s,  -x*s + y*c

def horiz_speed_bearing(v_ned):
    vn, ve, *_ = v_ned
    brg = float((np.arctan2(ve, vn) + 2*np.pi) % (2*np.pi))
    return brg
live_lats, live_lons = [], []
# ───── resample helper (GPS path ➜ model time grid) ────────────────────────
def resample_xy_to_t(xp: np.ndarray,
                     yp: np.ndarray,
                     t_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return xp/yp sampled at every t_model (33 pts)."""
    if len(xp) < 2:
        return np.zeros_like(t_model), np.zeros_like(t_model)

    seg_d = np.hypot(np.diff(xp), np.diff(yp))
    cum_d = np.insert(np.cumsum(seg_d), 0, 0.0)
    s_orig   = cum_d / cum_d[-1]
    s_target = (t_model - t_model[0]) / (t_model[-1] - t_model[0])

    xp_r = np.interp(s_target, s_orig, xp)
    yp_r = np.interp(s_target, s_orig, yp)
    return xp_r, yp_r

# ────── load reference route ───────────────────────────────────────────────
route_lats, route_lons = [], []
if args.route.suffix == ".csv":
    with args.route.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            lat = row.get("lat") or row.get("latitude")
            lon = row.get("lon") or row.get("longitude")
            if lat and lon:
                route_lats.append(float(lat)); route_lons.append(float(lon))
else:  # rlog preload
    try:
        from tools.lib.logreader import LogReader
    except ImportError:
        sys.exit("Need tools.lib.logreader for rlog preload")
    for ev in LogReader(str(args.route)):
        if ev.which() != "liveLocationKalmanDEPRECATED":
            continue
        g = ev.liveLocationKalmanDEPRECATED
        if g.status != "valid" or not g.positionGeodetic.valid:
            continue
        lat, lon, *_ = g.positionGeodetic.value
        route_lats.append(lat); route_lons.append(lon)

if not route_lats:
    sys.exit("No GPS points in route file.")
route_lats, route_lons = map(np.asarray, (route_lats, route_lons))
origin_lat, origin_lon = route_lats[0], route_lons[0]
print(f"Loaded {len(route_lats)} route points from {args.route}")

# ────────── Matplotlib boiler-plate ────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
fig.canvas.manager.set_window_title("GPS-centred + modelV2 override")

route_line, = ax.plot([], [], ':',  color="grey", alpha=.5, label="reference")
pred_line,  = ax.plot([], [], lw=3,  color="tab:blue",   label=f"next {args.n}")
model_line, = ax.plot([], [], '--', color="tab:purple", label="modelV2 path")
curr_pt,    = ax.plot([], [], "ro", ms=6, label="current")
heading_q   = ax.quiver([], [], [], [],
                        angles='xy', scale_units='xy', scale=1,
                        color='limegreen', width=0.012, label="heading")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_aspect("equal", adjustable="box"); ax.grid(True); ax.legend()
W = args.window

fig.canvas.draw()               # first paint
fig.canvas.flush_events()


PROCESS_NAME = "selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

VISION_ONNX_PATH = Path(__file__).parent / 'models/driving_vision.onnx'
POLICY_ONNX_PATH = Path(__file__).parent / 'models/driving_policy_aug.onnx'
VISION_METADATA_PATH = Path(__file__).parent / 'models/driving_vision_metadata.pkl'
POLICY_METADATA_PATH = Path(__file__).parent / 'models/driving_policy_metadata.pkl'

dv_onnx = ort.InferenceSession(VISION_ONNX_PATH, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
# dp_onnx = ort.InferenceSession(POLICY_ONNX_PATH, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
input_name = dv_onnx.get_inputs()[0].name
_, c_in, h_in, w_in = dv_onnx.get_inputs()[0].shape

device = torch.device("cuda")         # or "cpu"
# recreate the network the same way you did originally
path_to_onnx_vision_model = '/home/adas/openpilot/selfdrive/modeld/models/driving_vision_clean.onnx'
model_vision = onnx.load(path_to_onnx_vision_model)
pytorch_vision_model_loaded = onnx2pytorch.ConvertModel(model_vision).to(device)
pytorch_vision_model_loaded.requires_grad_(False)
pytorch_vision_model_loaded.eval()

path_to_onnx_policy_model = '/home/adas/openpilot/selfdrive/modeld/models/driving_policy_aug.onnx'
model_policy = onnx.load(path_to_onnx_policy_model)
pytorch_policy_model_loaded = onnx2pytorch.ConvertModel(model_policy).to(device)
pytorch_policy_model_loaded.requires_grad_(False)
pytorch_policy_model_loaded.eval()

# state_dict = torch.load(ckpt_path, map_location=device)
# pytorch_model_loaded.load_state_dict(state_dict, strict=True)

LAT_SMOOTH_SECONDS = 0.0
LONG_SMOOTH_SECONDS = 0.0
MIN_LAT_CONTROL_SPEED = 0.3

save_dir = "./saved_frames"
os.makedirs(save_dir, exist_ok=True)

def b(a):
    k = 0.01
    p = 4
    if a <= 16:
        return 0.1 * np.exp(-k * (16 - a)**p)
    else:
        return 0
from typing import Dict
def join_policy_outputs(
        heads: Dict[str, np.ndarray],
        slices: Dict[str, slice],
) -> np.ndarray:
    """
    Reverse of `slice_outputs`.  Works with negative starts (e.g. pad = -2).
    Returns shape (1, N).
    """
    # 1 ── total length ------------------------------------------------------
    pos_stop_max  = 0        # largest positive stop we see
    tail_required = 0        # how many extra elements negative slices need

    for k, sl in slices.items():
        if k not in heads:          # optional head not present
            continue
        size = heads[k].size

        start = 0 if sl.start is None else sl.start
        stop  = sl.stop

        if start >= 0:
            pos_stop_max = max(pos_stop_max, start + size)
        else:                       # negative start ⇒ counts from the end
            # example: start = -2, size = 2  →  need 2 elems of tail
            tail_required = max(tail_required, -start)

        if stop and stop > 0:       # rare case: slice(-3, -1)
            pos_stop_max = max(pos_stop_max, stop)

    flat_len = pos_stop_max + tail_required

    # 2 ── allocate ---------------------------------------------------------
    sample = next(iter(heads.values()))
    flat   = np.empty(flat_len, dtype=sample.dtype)

    # 3 ── fill -------------------------------------------------------------
    for k, sl in slices.items():
        if k not in heads:
            continue
        head  = heads[k].ravel()
        start = 0 if sl.start is None else sl.start
        if start < 0:
            start = flat_len + start      # convert to absolute index
        flat[start:start + head.size] = head

    return flat[np.newaxis, :]            # add batch dim

def get_action_from_model(model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                          lat_action_t: float, long_action_t: float, v_ego: float,distance:float) -> log.ModelDataV2.Action:
    plan = model_output['plan'][0]
    desired_accel, should_stop = get_accel_from_plan(plan[:,Plan.VELOCITY][:,0],
                                                     plan[:,Plan.ACCELERATION][:,0],
                                                     ModelConstants.T_IDXS,
                                                     action_t=long_action_t)
    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)
    desired_curvature = model_output['desired_curvature'][0, 0]

    print(f'model_desired_curvature:{desired_curvature}')
    # desired_curvature = max(min(model_output['desired_curvature'][0, 0],0.1),-0.1)
    if distance < 20:
      desired_curvature = 0.1
    if v_ego > MIN_LAT_CONTROL_SPEED:
      # cloudlog.warning(f'desired_curvature:{desired_curvature:.4f}')
      desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, LAT_SMOOTH_SECONDS)
    else:
      cloudlog.warning(f'ppppppppppppppppppppppppppppppppp:{desired_curvature}')
      desired_curvature = prev_action.desiredCurvature
    # cloudlog.warning(f'ppppppppppppppppppppppppppppppppp:{desired_curvature}')
    return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                  desiredAcceleration=float(desired_accel),
                                  shouldStop=bool(should_stop))
import matplotlib.pyplot as plt
def show_two_yuv_combined(raw: np.ndarray):
    """
    raw: uint8 array of shape (1,12,128,256)
    Decodes two YUV frames, stacks them horizontally, and displays in one window.
    """
    # 1) Remove batch and split into two 6-channel frames
    arr    = raw.squeeze(0)               # → (12,128,256)
    frames = arr.reshape(2, 6, 128, 256)  # → (2,6,128,256)

    bgr_frames = []
    for f in frames:
        # Rebuild full-res Y
        Y = np.zeros((256, 512), dtype=np.uint8)
        Y[0::2,0::2] = f[0]; Y[0::2,1::2] = f[1]
        Y[1::2,0::2] = f[2]; Y[1::2,1::2] = f[3]
        # Upsample U/V
        U = cv2.resize(f[4], (512,256), interpolation=cv2.INTER_LINEAR)
        V = cv2.resize(f[5], (512,256), interpolation=cv2.INTER_LINEAR)
        # Stack & convert
        yuv = np.stack((Y, U, V), axis=-1)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        bgr_frames.append(bgr)
        # bgr_frames = bgr

    # 2) Combine horizontally: result shape = (256, 512*2, 3)
    combined = np.hstack(bgr_frames)

    return combined

def show_all_feeds(image_feed: dict):
    """
    image_feed: dict[key -> BGR numpy array of shape (256, 1024, 3)]
    Stacks them vertically and shows in one window.
    """
    # make sure we have at least one feed
    if not image_feed:
        return

    # grab the list of arrays in insertion order
    frames = list(image_feed.values())

    # verify they all have the same width and channels
    h, w, c = frames[0].shape
    for f in frames:
        assert f.shape[1] == w and f.shape[2] == c, "All feeds must have same width & channels"

    # stack vertically into shape (256 * N, 1024, 3)
    canvas = np.vstack(frames)

    cv2.imshow("All Feeds", canvas)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     return


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof

class ModelState:
  frames: dict[str, DrivingModelFrame]
  inputs: dict[str, np.ndarray]
  output: np.ndarray
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, context: CLContext):
    self.frames = {
      'input_imgs': DrivingModelFrame(context, ModelConstants.TEMPORAL_SKIP),
      'big_input_imgs': DrivingModelFrame(context, ModelConstants.TEMPORAL_SKIP)
    }
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    nav_features = np.zeros(ModelConstants.NAV_FEATURE_LEN, dtype=np.float32)
    nav_instructions = np.zeros(ModelConstants.NAV_INSTRUCTION_LEN, dtype=np.float32)
    # print(nav_instructions)

    self.full_features_buffer = np.zeros((1, ModelConstants.FULL_HISTORY_BUFFER_LEN,  ModelConstants.FEATURE_LEN), dtype=np.float32)
    self.full_desire = np.zeros((1, ModelConstants.FULL_HISTORY_BUFFER_LEN, ModelConstants.DESIRE_LEN), dtype=np.float32)
    self.full_prev_desired_curv = np.zeros((1, ModelConstants.FULL_HISTORY_BUFFER_LEN, ModelConstants.PREV_DESIRED_CURV_LEN), dtype=np.float32)
    self.temporal_idxs = slice(-1-(ModelConstants.TEMPORAL_SKIP*(ModelConstants.INPUT_HISTORY_BUFFER_LEN-1)), None, ModelConstants.TEMPORAL_SKIP)
    self.dv_onnx = ort.InferenceSession(VISION_ONNX_PATH, providers=["CUDAExecutionProvider","CPUExecutionProvider"])

    # self.dp_onnx = ort.InferenceSession(POLICY_ONNX_PATH, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    self.pytorch_vision_model_loaded = pytorch_vision_model_loaded
    self.pytorch_policy_model_loaded = pytorch_policy_model_loaded

    with open(VISION_METADATA_PATH, 'rb') as f:
      vision_metadata = pickle.load(f)
      self.vision_input_shapes =  vision_metadata['input_shapes']
      self.vision_output_slices = vision_metadata['output_slices']
      vision_output_size = vision_metadata['output_shapes']['outputs'][1]

    with open(POLICY_METADATA_PATH, 'rb') as f:
      policy_metadata = pickle.load(f)
      self.policy_input_shapes =  policy_metadata['input_shapes']
      self.policy_output_slices = policy_metadata['output_slices']
      policy_output_size = policy_metadata['output_shapes']['outputs'][1]

    # img buffers are managed in openCL transform code
    self.vision_inputs: dict[str, Tensor] = {}
    self.vision_output = np.zeros(vision_output_size, dtype=np.float32)
    self.numpy_inputs = {
      'desire': np.zeros((1, ModelConstants.INPUT_HISTORY_BUFFER_LEN, ModelConstants.DESIRE_LEN), dtype=np.float16),
      'nav_features' : np.ones((1,ModelConstants.NAV_FEATURE_LEN), dtype=np.float32),
      'nav_instructions' : np.ones((1,ModelConstants.NAV_INSTRUCTION_LEN), dtype=np.float32),
      'traffic_convention': np.zeros((1, ModelConstants.TRAFFIC_CONVENTION_LEN), dtype=np.float16),
      'lateral_control_params': np.zeros((1, ModelConstants.LATERAL_CONTROL_PARAMS_LEN), dtype=np.float16),
      'prev_desired_curv': np.zeros((1, ModelConstants.INPUT_HISTORY_BUFFER_LEN, ModelConstants.PREV_DESIRED_CURV_LEN), dtype=np.float16),
      'features_buffer': np.zeros((1, ModelConstants.INPUT_HISTORY_BUFFER_LEN,  ModelConstants.FEATURE_LEN), dtype=np.float16),
    }
    self.policy_inputs = {k: Tensor(v, device='cuda').realize() for k,v in self.numpy_inputs.items()}
    self.policy_output = np.zeros(policy_output_size, dtype=np.float32)
    self.parser = Parser()


  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
    return parsed_model_outputs

  def slice_len(self,model_len,sl):
    if isinstance(sl, slice):
        # Pythonic interpretation of slice.start / slice.stop
        start = sl.start if sl.start is not None else 0
        if start < 0:
            start += model_len
        stop  = sl.stop  if sl.stop  is not None else model_len
        if stop  < 0:
            stop  += model_len
        return max(stop - start, 0)
    else:
        return len(sl)

  def run(self, buf: VisionBuf, wbuf: VisionBuf, transform: np.ndarray, transform_wide: np.ndarray,
                inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:

    inputs['desire'][0] = 0
    new_desire = np.where(inputs['desire'] - self.prev_desire > .99, inputs['desire'], 0)
    self.prev_desire[:] = inputs['desire']
    self.full_desire[0,:-1] = self.full_desire[0,1:]
    self.full_desire[0,-1] = new_desire
    # print(f'new_desire:{new_desire}')
    self.numpy_inputs['desire'][:] = self.full_desire.reshape((1,ModelConstants.INPUT_HISTORY_BUFFER_LEN,ModelConstants.TEMPORAL_SKIP,-1)).max(axis=2)
    # print(self.numpy_inputs['desire'][:])
    self.numpy_inputs['traffic_convention'][:] = inputs['traffic_convention']
    self.numpy_inputs['lateral_control_params'][:] = inputs['lateral_control_params']
    onnx_feed = {}
    image_feed = {}
    torch_feed = {}
    # print(transform_wide)
    imgs_cl = {'input_imgs': self.frames['input_imgs'].prepare(buf, transform.flatten()),
               'big_input_imgs': self.frames['big_input_imgs'].prepare(wbuf, transform_wide.flatten())}

    for key in imgs_cl:
      frame_input = self.frames[key].buffer_from_cl(imgs_cl[key]).reshape(self.vision_input_shapes[key])
      self.vision_inputs[key] = Tensor(frame_input, dtype=dtypes.uint8).realize()
      tensor = self.vision_inputs[key]
      if isinstance(tensor, Tensor):
          np_arr = tensor.numpy()
      else:
          np_arr = tensor
      assert np_arr.dtype == np.uint8, f"{key} is {np_arr.dtype}, expected uint8"
      onnx_feed[key] = np_arr
      torch_tensor = torch.from_numpy(np_arr).to(device)

      torch_feed[key] = torch_tensor                              # add to dict
      image_feed[key] = show_two_yuv_combined(np_arr)

    # show_all_feeds(image_feed)

    onnx_outputs = self.dv_onnx.run(None, onnx_feed)


    with torch.inference_mode():
      pytorch_outputs = self.pytorch_vision_model_loaded(**torch_feed)
    pytorch_outputs = [out.detach().cpu().numpy() for out in pytorch_outputs]

    output_array = onnx_outputs[0].reshape(-1)
    model_len_v    = output_array.shape[0]

    vision_outputs_dict = self.parser.parse_vision_outputs(self.slice_outputs(output_array, self.vision_output_slices))
    self.full_features_buffer[0,:-1] = self.full_features_buffer[0,1:]
    self.full_features_buffer[0,-1] = vision_outputs_dict['hidden_state'][0, :]
    self.numpy_inputs['features_buffer'][:] = self.full_features_buffer[0, self.temporal_idxs]

    torch_policy_inputs = {
      name: torch.from_numpy(arr).to(device).to(torch.float16)
      for name, arr in self.numpy_inputs.items()
    }
    with torch.inference_mode():
      pytorch_p_outputs = self.pytorch_policy_model_loaded(**torch_policy_inputs)
    pytorch_p_outputs = [out.detach().cpu().numpy() for out in pytorch_p_outputs]


    self.policy_ouput = pytorch_p_outputs
    print(self.policy_output_slices)

    policy_outputs_dict = self.parser.parse_policy_outputs(self.slice_outputs(self.policy_ouput[0].reshape(-1), self.policy_output_slices))
    print(self.slice_outputs(self.policy_ouput[0].reshape(-1), self.policy_output_slices))
    print(policy_outputs_dict)
    combined_outputs_dict = {**vision_outputs_dict, **policy_outputs_dict}
    # print(policy_outputs_dict.keys())
    heads     = policy_outputs_dict
    flat_reco = join_policy_outputs(heads, self.policy_output_slices)

    print(len(self.policy_ouput[0]))
    print(len(flat_reco[0]))
    # assert flat_reco.shape ==
    policy_outputs_dict = self.parser.parse_policy_outputs(self.slice_outputs(flat_reco[0].reshape(-1), self.policy_output_slices))
    # combined_outputs_dict = {**vision_outputs_dict, **policy_outputs_dict}

    return combined_outputs_dict,image_feed

from collections import namedtuple
def bgr_to_nv12_buf(bgr: np.ndarray) -> VisionBuf:
    """
    Convert a BGR image to an NV12 raw buffer packed into a VisionBuf-like namedtuple.
    """
    h, w = bgr.shape[:2]
    # NV12 expects stride = width (you may need to pad to alignment in real use)
    stride = w
    uv_offset = h * stride

    # 2. Convert BGR → YUV420p (I420 planar) via OpenCV
    #    This gives a flat array: [Y plane (h*w), U plane (h/2*w/2), V plane (h/2*w/2)]
    yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).flatten()

    # 3. Extract Y, U, V planes
    y_size = h * w
    uv_half_size = (h // 2) * (w // 2)
    y_plane = yuv_i420[0 : y_size].reshape((h, w))
    u_plane = yuv_i420[y_size : y_size + uv_half_size].reshape((h // 2, w // 2))
    v_plane = yuv_i420[y_size + uv_half_size : y_size + 2*uv_half_size].reshape((h // 2, w // 2))

    # 4. Build the interleaved UV plane for NV12 (semi-planar):
    #    for each row r, each column c: uv_row[2*c] = U[r,c], uv_row[2*c+1] = V[r,c]
    uv_plane = np.empty((h // 2, w), dtype=np.uint8)
    uv_plane[:, 0::2] = u_plane
    uv_plane[:, 1::2] = v_plane

    # 5. Stack Y and UV into one NV12 buffer (shape (h + h/2, w))
    nv12 = np.vstack((y_plane, uv_plane))

    # 6. Package into VisionBuf: .data should be bytes or a buffer object
    return VisionBuf(
        data=nv12.tobytes(),
        height=h,
        width=w,
        stride=stride,
        uv_offset=uv_offset
    )





def main(demo=False):
  # cv2.namedWindow("All Feeds", cv2.WINDOW_NORMAL)
  cloudlog.warning("modeld init")
  from cereal.services import SERVICE_LIST   # modern branches
# from cereal import service_list         # very old branches

  ALL_TOPICS = [srv[0] for srv in SERVICE_LIST]   # first field is the name
  print(f"{len(ALL_TOPICS)} services: ", ALL_TOPICS)


  sentry.set_tag("daemon", PROCESS_NAME)
  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)
  if not USBGPU:
    # USB GPU currently saturates a core so can't do this yet,
    # also need to move the aux USB interrupts for good timings
    config_realtime_process(7, 54)

  cloudlog.warning("setting up CL context")
  cl_context = CLContext()
  cloudlog.warning("CL context ready; loading model")
  model = ModelState(cl_context)
  cloudlog.warning("models loaded, modeld starting")

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, cl_context)
  # vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, cl_context)

  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, cl_context)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  # messaging
  pm = PubMaster([ "modelV2","drivingModelData", "cameraOdometry"])
  sm = SubMaster([ "deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "liveDelay","navModelDEPRECATED", "navInstruction","liveLocationKalmanDEPRECATED","gnssMeasurements"])

  publish_state = PublishState()
  params = Params()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()


  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("modeld got CarParams: %s", CP.brand)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  # TODO Move smooth seconds to action function
  long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
  prev_action = log.ModelDataV2.Action()

  DH = DesireHelper()
  frame_idx = 0
  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      attrs = [a for a in dir(buf_main) if not a.startswith('_')]
      # cloudlog.warning(f"VisionBuf attrs: {attrs}")
      if buf_main is None:
        break

      h         = buf_main.height
      w         = buf_main.width
      s         = buf_main.stride     # bytes per row in both Y & UV planes
      uv_off    = buf_main.uv_offset  # byte offset where UV plane starts in buf.data

      # 2. View the entire buffer as one flat uint8 array
      raw = np.frombuffer(buf_main.data, dtype=np.uint8)

      # 3. Extract & reshape the Y plane (h rows × s bytes), then crop to actual width
      y_plane = raw[0 : h*s]\
                  .reshape((h, s))\
                  [:, :w]

      # 4. Extract & reshape the UV plane ((h/2) rows × s bytes), then crop
      uv_plane = raw[uv_off : uv_off + (h//2)*s]\
                  .reshape((h//2, s))\
                  [:, :w]

      # 5. Stack into NV12 layout and convert to BGR
      nv12 = np.vstack((y_plane, uv_plane))
      bgr  = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

      resized = cv2.resize(bgr, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
      # 2. BGR → RGB
      rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
      # 3. Normalize to [0,1]
      img_f32 = rgb.astype(np.float32) / 255.0
      # 4. HWC → CHW, add batch dim
      model_input = img_f32.transpose(2, 0, 1)[None, :, :, :]  # shape (1,3,h_in,w_in)

      # 5. Run inference
      # outputs = dv_onnx.run(None, {input_name: model_input})
      #    → `outputs` is a list of numpy arrays, one per output of your onnx graph

      # 6. Display
      # cv2.imshow("road camera", bgr)
      # cv2.waitKey(1)
      filename = os.path.join(save_dir, f"frame_{meta_main.timestamp_sof}.png")
      success = cv2.imwrite(filename, bgr)
      if not success:
          print(f"Failed to write {filename}")
      frame_idx += 1


    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break
        h         = buf_extra.height
        w         = buf_extra.width
        s         = buf_extra.stride     # bytes per row in both Y & UV planes
        uv_off    = buf_extra.uv_offset  # byte offset where UV plane starts in buf.data

        # 2. View the entire buffer as one flat uint8 array
        raw = np.frombuffer(buf_extra.data, dtype=np.uint8)

        # 3. Extract & reshape the Y plane (h rows × s bytes), then crop to actual width
        y_plane = raw[0 : h*s]\
                    .reshape((h, s))\
                    [:, :w]

        # 4. Extract & reshape the UV plane ((h/2) rows × s bytes), then crop
        uv_plane = raw[uv_off : uv_off + (h//2)*s]\
                    .reshape((h//2, s))\
                    [:, :w]

        # 5. Stack into NV12 layout and convert to BGR
        nv12 = np.vstack((y_plane, uv_plane))
        bgr  = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

        resized = cv2.resize(bgr, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        # 2. BGR → RGB
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 3. Normalize to [0,1]
        img_f32 = rgb.astype(np.float32) / 255.0
        # 4. HWC → CHW, add batch dim
        model_input = img_f32.transpose(2, 0, 1)[None, :, :, :]  # shape (1,3,h_in,w_in)

        # 5. Run inference
        # outputs = dv_onnx.run(None, {input_name: model_input})
        #    → `outputs` is a list of numpy arrays, one per output of your onnx graph

        # 6. Display
        # cv2.imshow("extra camera", bgr)
        cv2.waitKey(1)

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                         extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)

    g = sm["liveLocationKalmanDEPRECATED"]

    if g.status != "valid" or not g.positionGeodetic.valid:
        return

    lat, lon, _ = g.positionGeodetic.value
    live_lats.append(lat); live_lons.append(lon)

    # bearing
    if g.velocityNED.valid:
        bearing = horiz_speed_bearing(g.velocityNED.value)
    elif len(live_lats) > 1:
        _gx, _gy = equirect(np.array(live_lats[-2:]),
                            np.array(live_lons[-2:]),
                            origin_lat, origin_lon)
        dx, dy = _gx[-1]-_gx[-2], _gy[-1]-_gy[-2]
        bearing = np.arctan2(dx, dy) if (dx or dy) else 0.0
    else:
        bearing = 0.0

    # look-ahead slice
    dists = haversine(route_lats, route_lons, lat, lon)
    idx   = int(np.argmin(dists))
    nxt_lat = route_lats[idx:idx+args.n] if dists[idx] < args.thresh else np.array([])
    nxt_lon = route_lons[idx:idx+args.n] if dists[idx] < args.thresh else np.array([])

    # planar projection
    gx,  gy  = equirect(np.asarray(live_lats), np.asarray(live_lons),
                        origin_lat, origin_lon)
    grx, gry = equirect(route_lats, route_lons, origin_lat, origin_lon)
    gxp, gyp = equirect(nxt_lat, nxt_lon, origin_lat, origin_lon)

    # translate so car at (0,0)
    tx, ty = gx[-1], gy[-1]
    gx  -= tx; gy  -= ty
    grx -= tx; gry -= ty
    gxp -= tx; gyp -= ty

    # rotate to heading-up
    x,  y  = rotate(gx,  gy,  -bearing)
    rx, ry = rotate(grx, gry, -bearing)
    xp, yp = rotate(gxp, gyp, -bearing) if len(gxp) else (np.array([]), np.array([]))


    route_line.set_data(rx, ry)
    # pred_line .set_data(xp, yp)
    curr_pt   .set_data([0], [0])
    heading_q .set_offsets([[0, 0]]); heading_q.set_UVC([0], [10])

    ax.set_xlim(-W, W); ax.set_ylim(-W, W)
    ax.set_title(f"Fixes {len(live_lats)} | match {dists[idx]:.1f} m")

    fig.canvas.draw_idle()       # schedule a redraw
    fig.canvas.flush_events()    # actually push to the screen
    # plt.pause(0.001)


    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    # print(sm["navInstruction"])
    # print(sm["navModelDEPRECATED"])
    print(sm["liveLocationKalmanDEPRECATED"].positionGeodetic)
    print(sm["liveLocationKalmanDEPRECATED"].positionECEF)
    next_maneuver = {
      'type': sm["navInstruction"].maneuverType,
      'modifier': sm["navInstruction"].maneuverModifier,
      'street': sm["navInstruction"].maneuverPrimaryText,
      'distance': sm["navInstruction"].maneuverDistance
    }

    # Displaying the next maneuver
    if next_maneuver['distance'] != 0:
     print(f"Next maneuver: {next_maneuver['type']} {next_maneuver['modifier']} onto {next_maneuver['street']} in {next_maneuver['distance']:.2f} meters. desire_:{1/(4*next_maneuver['distance']/(2*np.pi))}")

    lat_delay = sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS
    cloudlog.warning(f'steeringAngleDeg:{sm["carState"].steeringAngleDeg}')
    cloudlog.warning(f'steeringTorque:{sm["carState"].steeringTorque}')
    cloudlog.warning(f'steeringPressed:{sm["carState"].steeringPressed}')
    # cloudlog.warning(f'Model_desire_curv:{sm["modelV2"].action.desiredCurvature}')
    lateral_control_params = np.array([v_ego, lat_delay], dtype=np.float32)
    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      # print(f'dc:{dc}')
      model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics, True).astype(np.float32)
      # model_transform_extra = np.array([
      #     [2 , 0,   6.00883545e+02    ],
      #     [0, 2 ,   3.62377625e+02     ],
      #     [0, 0, 1]
      # ], dtype=np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1
    H = model_transform_extra  # shape (3,3), dtype float32
    t_idxs = np.array(ModelConstants.T_IDXS, dtype=np.float32)


    theta  = -np.pi/2                          # –90 degrees in radians
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[ cos_t, -sin_t],
                  [ sin_t,  cos_t]], dtype=np.float32)

    def draw_model_path(canvas, x_coeffs, y_coeffs):
        h, w = canvas.shape[:2]
        bottom_center = np.array([w//2, h-1], dtype=np.float32)

        # 1) Evaluate in model-frame
        xs = polyval(t_idxs, x_coeffs)
        ys = polyval(t_idxs, y_coeffs)
        # print(max(xs))
        # print(max(ys))
        # 2) Project through homography
        pts_model = np.stack([xs, ys, np.ones_like(xs)], axis=0)  # (3,N)
        pts_img   = H.dot(pts_model)                              # (3,N)
        pts_img  /= pts_img[2:3, :]                               # normalize
        pts2d     = pts_img[:2, :].T                              # (N,2)

        # 3) Shift so first point is at bottom-center
        shift = bottom_center - pts2d[0]
        pts2d = pts2d + shift

        # 4) Rotate all points by –90° about bottom-center
        centered = pts2d - bottom_center      # (N,2)
        rotated  = (R @ centered.T).T         # (N,2)
        pts2d_rot = rotated + bottom_center   # (N,2)

        # 5) Draw
        pts_int = pts2d_rot.reshape(-1,1,2).astype(np.int32)
        cv2.polylines(canvas, [pts_int], isClosed=False, color=(0,255,0), thickness=2)
    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0
    if prepare_only:
      cloudlog.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")

    inputs:dict[str, np.ndarray] = {
      'desire': vec_desire,
      'traffic_convention': traffic_convention,
      'lateral_control_params': lateral_control_params,
      }

    mt1 = time.perf_counter()
    model_output,canvas_feed = model.run(buf_main, buf_extra, model_transform_main, model_transform_extra, inputs)
    # model_ouput = None
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')
      print(model_output['desired_curvature'][0][0])

      # cloudlog.warning(1/model_output['desired_curvature'][0][0])
      # cloudlog.warning(model_output['lane_lines_prob'])
      # if model_output['desired_curvature'][0][0]>0.03:
      #   model_output['desired_curvature'][0][0] = 0.01
      action = get_action_from_model(model_output, prev_action, lat_delay + DT_MDL, long_delay + DT_MDL, v_ego,sm["navInstruction"].maneuverDistance)
      prev_action = action
      fill_model_msg(drivingdata_send, modelv2_send, model_output, action,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)
      # print(modelv2_send.modelV2.position)


      i = 9
      m_x = np.array(modelv2_send.modelV2.position.x)
      m_y = np.array(modelv2_send.modelV2.position.y)
      x0,x1,x2 = m_x[i-1:i+2]
      y0,y1,y2 = m_y[i-1:i+2]
      dx = x2-x0
      y_p = (y2-y0)/dx
      h = dx/2
      y_pp = (y2 - 2*y1 + y0)/h**2
      d_c = y_pp/(1+y_p**2)**1.5
      print(f"es_dc:{d_c}")
      if len(modelv2_send.modelV2.position.x):
            fwd  = np.array(modelv2_send.modelV2.position.x)        # forward (X) ➜ +Y after swap
            left = np.array(modelv2_send.modelV2.position.y)        # left (Y) ➜ +X after swap
            mdl_x, mdl_y = left, fwd
            model_line.set_data(mdl_x, mdl_y)
      tgrid = np.array(modelv2_send.modelV2.position.t)     # 33-pt grid

      xp_r, yp_r = resample_xy_to_t(xp.astype(float),
                                    yp.astype(float),
                                    tgrid)

      try:
        # Savitzky–Golay: odd window, poly-order < window
        from scipy.signal import savgol_filter

        win = 9                       # experiment: 5-9 points work well for 33-pt path
        win = win | 1                 # make sure it’s odd
        if len(tgrid) < win:          # short safety
            win = len(tgrid) | 1

        poly = 3                      # cubic fits most OP paths
        xp_smooth = savgol_filter(xp_r, win, poly)
        yp_smooth = savgol_filter(yp_r, win, poly)
        # print(xp_r)
        # print(yp_r)

      except ImportError:
        # Fallback: centred moving average (very light smoothing)
        k = 5                         # window size
        kernel = np.ones(k, dtype=float) / k
        xp_smooth = np.convolve(xp_r, kernel, mode='same')
        yp_smooth = np.convolve(yp_r, kernel, mode='same')
      pred_line .set_data(xp_smooth, yp_smooth)
      pos = modelv2_send.modelV2.position           # shortcut
      # print(f'pos_x:{pos.x}')
      print(f'pos:{pos}')

      pos.x     = yp_smooth.astype(float).tolist()       # forward → x
      pos.y     = xp_smooth.astype(float).tolist()       # left    → y
      pos.xStd  = [0.05] * len(tgrid)
      pos.yStd  = [0.05] * len(tgrid)





      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
      drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      # cloudlog.warning(f'modelv2_send:{modelv2_send}')
      # cloudlog.warning(f'drivingdata_send:{drivingdata_send}')
      # cloudlog.warning(f'posenet_send:{posenet_send}')
      x_coef = drivingdata_send.drivingModelData.path.xCoefficients  # length 5 list
      y_coef = drivingdata_send.drivingModelData.path.yCoefficients
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
      # draw_model_path(canvas_feed['big_input_imgs'], x_coef, y_coef)

    #   # show the result
    #   cv2.imshow("All Feeds with Path", canvas_feed['big_input_imgs'])
    #   if cv2.waitKey(1) & 0xFF == ord('q'):
          # break

    last_vipc_frame_id = meta_main.frame_id
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # plt.show()

if __name__ == "__main__":
  try:
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    # args = parser.parse_args()
    main()
    # cv2.destroyAllWindows()
  except KeyboardInterrupt:
    cloudlog.warning(f"child {PROCESS_NAME} got SIGINT")
  except Exception:
    sentry.capture_exception()
    raise
