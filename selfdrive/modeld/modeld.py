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
import pathlib
from tinygrad.tensor import Tensor
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
from extra.onnx import OnnxRunner
from tinygrad.nn.state import load_state_dict



PROCESS_NAME = "selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

VISION_ONNX_PATH = Path(__file__).parent / 'models/driving_vision.onnx'
POLICY_ONNX_PATH = Path(__file__).parent / 'models/driving_policy.onnx'
VISION_PKL_PATH = Path(__file__).parent / 'models/driving_vision.pkl'
POLICY_PKL_PATH = Path(__file__).parent / 'models/driving_policy.pkl'
VISION_METADATA_PATH = Path(__file__).parent / 'models/driving_vision_metadata.pkl'
POLICY_METADATA_PATH = Path(__file__).parent / 'models/driving_policy_metadata.pkl'

dv_onnx = ort.InferenceSession(VISION_ONNX_PATH, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
dp_onnx = ort.InferenceSession(POLICY_ONNX_PATH, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
input_name = dv_onnx.get_inputs()[0].name
_, c_in, h_in, w_in = dv_onnx.get_inputs()[0].shape

device = torch.device("cuda")         # or "cpu"
# recreate the network the same way you did originally
path_to_onnx_vision_model = '/home/adas/openpilot/selfdrive/modeld/models/driving_vision_clean.onnx'
model_vision = onnx.load(path_to_onnx_vision_model)
pytorch_vision_model_loaded = onnx2pytorch.ConvertModel(model_vision).to(device)
pytorch_vision_model_loaded.requires_grad_(False)
pytorch_vision_model_loaded.eval()

path_to_onnx_policy_model = '/home/adas/openpilot/selfdrive/modeld/models/driving_policy_with_nav.onnx'
model_policy = onnx.load(path_to_onnx_policy_model)
pytorch_policy_model_loaded = onnx2pytorch.ConvertModel(model_policy).to(device)
pytorch_policy_model_loaded.requires_grad_(False)
pytorch_policy_model_loaded.eval()


# model_path = "/home/adas/openpilot/selfdrive/modeld/models/driving_vision_clean.onnx"
# runner      = OnnxRunner(onnx.load(model_path))       # runner is already callable

# # --- hydrate weights -------------------------------------------------------
# with open("/home/adas/openpilot/selfdrive/modeld/models/driving_vision_tinygrad.pkl", "rb") as f:
#     saved_state = pickle.load(f)
# load_state_dict(runner, saved_state)

# model_path = "/home/adas/openpilot/selfdrive/modeld/models/driving_policy.onnx"
# p_runner      = OnnxRunner(onnx.load(model_path))       # runner is already callable

# # --- hydrate weights -------------------------------------------------------
# with open("/home/adas/openpilot/selfdrive/modeld/models/driving_policy_tinygrad.pkl", "rb") as f:
#     saved_state = pickle.load(f)
# load_state_dict(p_runner, saved_state)
# state_dict = torch.load(ckpt_path, map_location=device)
# pytorch_model_loaded.load_state_dict(state_dict, strict=True)

LAT_SMOOTH_SECONDS = 0.0
LONG_SMOOTH_SECONDS = 0.0
MIN_LAT_CONTROL_SPEED = 0.3

save_dir = "./saved_frames"
os.makedirs(save_dir, exist_ok=True)

from tinygrad import TinyJit
# @TinyJit
# def vision_run(inputs):
#     # OpenPilot expects the first (and only) output flattened
#     return runner(inputs)['outputs']

def b(a):
    k = 0.01
    p = 4
    if a <= 16:
        return 0.1 * np.exp(-k * (16 - a)**p)
    else:
        return 0

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
      cloudlog.warning(f'desired_curvature:{desired_curvature:.4f}')
      desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, LAT_SMOOTH_SECONDS)
    else:
      cloudlog.warning(f'ppppppppppppppppppppppppppppppppp:{desired_curvature}')
      desired_curvature = prev_action.desiredCurvature
    # cloudlog.warning(f'ppppppppppppppppppppppppppppppppp:{desired_curvature}')
    return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                  desiredAcceleration=float(desired_accel),
                                  shouldStop=bool(should_stop))
import matplotlib.pyplot as plt

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
    # self.dv_onnx = ort.InferenceSession(VISION_ONNX_PATH, providers=["CUDAExecutionProvider","CPUExecutionProvider"])

    # self.dp_onnx = ort.InferenceSession(POLICY_ONNX_PATH, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    # self.pytorch_vision_model_loaded = pytorch_vision_model_loaded
    # self.pytorch_policy_model_loaded = pytorch_policy_model_loaded

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

    with open(VISION_PKL_PATH, "rb") as f:
      self.vision_run = pickle.load(f)

    with open(POLICY_PKL_PATH, "rb") as f:
      self.policy_run = pickle.load(f)

    # img buffers are managed in openCL transform code
    self.vision_inputs: dict[str, Tensor] = {}
    self.vision_output = np.zeros(vision_output_size, dtype=np.float32)
    self.numpy_inputs = {
      'desire': np.zeros((1, ModelConstants.INPUT_HISTORY_BUFFER_LEN, ModelConstants.DESIRE_LEN), dtype=np.float32),
      'traffic_convention': np.zeros((1, ModelConstants.TRAFFIC_CONVENTION_LEN), dtype=np.float32),
      'lateral_control_params': np.zeros((1, ModelConstants.LATERAL_CONTROL_PARAMS_LEN), dtype=np.float32),
      'prev_desired_curv': np.zeros((1, ModelConstants.INPUT_HISTORY_BUFFER_LEN, ModelConstants.PREV_DESIRED_CURV_LEN), dtype=np.float32),
      'features_buffer': np.zeros((1, ModelConstants.INPUT_HISTORY_BUFFER_LEN,  ModelConstants.FEATURE_LEN), dtype=np.float32),
    }
    self.policy_inputs = {k: Tensor(v, device='NPY').realize() for k,v in self.numpy_inputs.items()}
    self.policy_output = np.zeros(policy_output_size, dtype=np.float32)
    self.parser = Parser()


  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
    return parsed_model_outputs


  def run(self, buf: VisionBuf, wbuf: VisionBuf, transform: np.ndarray, transform_wide: np.ndarray,
                inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
      new_desire                  = np.where(inputs["desire"] - self.prev_desire > 0.99,
                                            inputs["desire"], 0)
      self.prev_desire[:]         = inputs["desire"]
      self.full_desire[0, :-1]    = self.full_desire[0, 1:]
      self.full_desire[0, -1]     = new_desire
      self.numpy_inputs["desire"][:] = self.full_desire.reshape(
          1, ModelConstants.INPUT_HISTORY_BUFFER_LEN, ModelConstants.TEMPORAL_SKIP, -1
      ).max(axis=2)

      self.numpy_inputs["traffic_convention"][:]   = inputs["traffic_convention"]
      self.numpy_inputs["lateral_control_params"][:] = inputs["lateral_control_params"]

      imgs_cl = {'input_imgs': self.frames['input_imgs'].prepare(buf, transform.flatten()),
                'big_input_imgs': self.frames['big_input_imgs'].prepare(wbuf, transform_wide.flatten())}

      for key in imgs_cl:
        frame_input = self.frames[key].buffer_from_cl(imgs_cl[key]).reshape(self.vision_input_shapes[key])
        self.vision_inputs[key] = Tensor(frame_input, dtype=dtypes.uint8,  device='GPU').realize()

      vision_out = self.vision_run(**self.vision_inputs).numpy().ravel()

      vision_dict = self.parser.parse_vision_outputs(
          self.slice_outputs(vision_out, self.vision_output_slices)
      )

      self.full_features_buffer[0, :-1] = self.full_features_buffer[0, 1:]
      self.full_features_buffer[0, -1]  = vision_dict["hidden_state"][0]
      self.numpy_inputs["features_buffer"][:] = \
          self.full_features_buffer[0, self.temporal_idxs]

      policy_out   = self.policy_run(**self.policy_inputs).numpy().ravel()

      policy_dict = self.parser.parse_policy_outputs(
          self.slice_outputs(policy_out, self.policy_output_slices)
      )

      return {**vision_dict, **policy_dict}, {}


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
  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      attrs = [a for a in dir(buf_main) if not a.startswith('_')]
      # cloudlog.warning(f"VisionBuf attrs: {attrs}")
      if buf_main is None:
        break


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
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    # print(sm["navInstruction"])
    # print(sm["navModelDEPRECATED"])
    # print(sm["liveLocationKalmanDEPRECATED"].positionGeodetic)
    # print(sm["liveLocationKalmanDEPRECATED"].positionECEF)
    # next_maneuver = {
    #   'type': sm["navInstruction"].maneuverType,
    #   'modifier': sm["navInstruction"].maneuverModifier,
    #   'street': sm["navInstruction"].maneuverPrimaryText,
    #   'distance': sm["navInstruction"].maneuverDistance
    # }

    # # Displaying the next maneuver
    # if next_maneuver['distance'] != 0:
    #  print(f"Next maneuver: {next_maneuver['type']} {next_maneuver['modifier']} onto {next_maneuver['street']} in {next_maneuver['distance']:.2f} meters. desire_:{1/(4*next_maneuver['distance']/(2*np.pi))}")

    lat_delay = sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS
    # cloudlog.warning(f'steeringAngleDeg:{sm["carState"].steeringAngleDeg}')
    # cloudlog.warning(f'steeringTorque:{sm["carState"].steeringTorque}')
    # cloudlog.warning(f'steeringPressed:{sm["carState"].steeringPressed}')
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
      action = get_action_from_model(model_output, prev_action, lat_delay + DT_MDL, long_delay + DT_MDL, v_ego,sm["navInstruction"].maneuverDistance)
      prev_action = action
      fill_model_msg(drivingdata_send, modelv2_send, model_output, action,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

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
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)


    last_vipc_frame_id = meta_main.frame_id


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
