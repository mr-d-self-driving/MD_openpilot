#!/usr/bin/env python3
"""
gps_display_kalman.py  •  live Kalman‑fused GPS plotting + desired curvature logging
=======================================================================

  ── Streams `liveLocationKalmanDEPRECATED` **and** `controlsState`      (Cereal)
  ── Live red dot & blue path                                (Matplotlib)
  ── Optional historical preload log    (--preload FILE.rlog.bz2)
  ── Adjustable refresh rate            (--hz 10)
  ── Continuous CSV logging (GPS + desired curvature)        (--out track.csv)

Example
-------
  # Just watch + save
  python gps_display_kalman.py

  # Preload an rlog, plot at 5 Hz, save to a named CSV
  python gps_display_kalman.py --preload 2024‑05‑12.rlog.bz2 --hz 5 --out may12_track.csv
"""
# --------------------------------------------------------------------------- #
#  Imports
# --------------------------------------------------------------------------- #
import argparse, csv, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cereal.messaging as messaging

try:                                   # optional for --preload
    from tools.lib.logreader import LogReader
except ImportError:
    LogReader = None


# --------------------------------------------------------------------------- #
#  Args
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--hz",       type=float, default=100, help="animation refresh rate (Hz)")
    p.add_argument("--preload",  type=Path, help="rlog / *.bz2 file to draw before going live")
    p.add_argument("--out",      type=Path, help="CSV file for storing fixes")
    p.add_argument("--name",     default="kalman_track_CS", help="stem for auto‑named CSV")
    return p

args = build_parser().parse_args()


# --------------------------------------------------------------------------- #
#  Output CSV setup
# --------------------------------------------------------------------------- #
if args.out:
    csv_path = args.out.expanduser()
else:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = Path(f"{args.name}_{stamp}.csv")

csv_path.parent.mkdir(parents=True, exist_ok=True)
csv_file   = csv_path.open("w", newline="", buffering=1)     # line‑buffered
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["unix_ms", "latitude", "longitude",
                     "altitude_m", "speed_mps", "bearing_deg",
                     "desired_curvature_radpm"])
print(f"Logging Kalman fixes ➜ {csv_path.resolve()}")


# --------------------------------------------------------------------------- #
#  Historical preload
# --------------------------------------------------------------------------- #
lats, lons = [], []

if args.preload and args.preload.exists():
    if LogReader is None:
        print("Preload requested but tools.lib.logreader not available; skipping.")
    else:
        print(f"Pre‑loading {args.preload} …")
        for ev in LogReader(str(args.preload)):
            if ev.which() != "liveLocationKalmanDEPRECATED":
                continue
            g = ev.liveLocationKalmanDEPRECATED
            if g.status != "valid" or not g.positionGeodetic.valid:
                continue
            lat, lon, *_ = g.positionGeodetic.value
            lats.append(lat);  lons.append(lon)
        print(f"  Loaded {len(lats)} points.")


# --------------------------------------------------------------------------- #
#  Projection helper (simple equirectangular)
# --------------------------------------------------------------------------- #

def equirect_xy(lat, lon, lat0, lon0):
    R = 6_378_137.0
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    x = R * np.cos(np.deg2rad(lat0)) * dlon
    y = R * dlat
    return x, y


# --------------------------------------------------------------------------- #
#  Live subscription
# --------------------------------------------------------------------------- #
sm = messaging.SubMaster(["liveLocationKalmanDEPRECATED", "controlsState"])
origin_latlon = [None, None]           # will hold first lat/lon

desired_curvature_latest = np.nan      # updated whenever controlsState arrives


# --------------------------------------------------------------------------- #
#  Matplotlib boiler‑plate
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6, 6))
fig.canvas.manager.set_window_title("Kalman GPS track (live)")

track_line, = ax.plot([], [], lw=2, color="tab:blue", label="path history")
curr_pt,   = ax.plot([], [], "ro", ms=6, label="current fix")

ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.legend(loc="upper left")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def horiz_speed_and_bearing(v_ned):
    """Return horizontal speed (m/s) and bearing deg (0°=N, 90°=E)"""
    vn, ve, *_ = v_ned
    speed = float(np.hypot(vn, ve))
    bearing_deg = (np.degrees(np.arctan2(ve, vn)) + 360.) % 360.
    return speed, bearing_deg


# --------------------------------------------------------------------------- #
#  Animation callback
# --------------------------------------------------------------------------- #

def update(_):
    global desired_curvature_latest

    sm.update(0)                       # non‑blocking poll

    # --- pull latest desired curvature if present ---------------------------
    if sm.updated["controlsState"]:
        cs = sm["controlsState"]
        # Field name changed across releases; try both
        if hasattr(cs, "curvatureDesired"):
            desired_curvature_latest = float(cs.curvatureDesired)
        elif hasattr(cs, "desiredCurvature"):
            desired_curvature_latest = float(cs.desiredCurvature)
        else:
            desired_curvature_latest = np.nan

    # --- handle Kalman fix ---------------------------------------------------
    if sm.updated["liveLocationKalmanDEPRECATED"]:
        g = sm["liveLocationKalmanDEPRECATED"]

        # validity guards
        if g.status != "valid" or not g.positionGeodetic.valid:
            return track_line, curr_pt

        lat, lon, alt = g.positionGeodetic.value
        if origin_latlon[0] is None:            # 1st fix becomes origin
            origin_latlon[:] = [lat, lon]

        lats.append(lat); lons.append(lon)

        # derive speed & bearing
        if g.velocityNED.valid:
            speed, bearing = horiz_speed_and_bearing(g.velocityNED.value)
        else:
            speed   = np.nan
            bearing = np.nan

        # write to CSV
        csv_writer.writerow([g.unixTimestampMillis,
                             lat, lon, alt,
                             speed, bearing,
                             desired_curvature_latest])
        if len(lats) % 100 == 0:
            print(f"  stored {len(lats)} fixes…")

        # update plot
        x, y = equirect_xy(np.array(lats), np.array(lons),
                           origin_latlon[0], origin_latlon[1])

        track_line.set_data(x, y)
        curr_pt.set_data(x[-1:], y[-1:])

        curv_txt = (f"{desired_curvature_latest:.4f}" if not np.isnan(desired_curvature_latest)
                     else "n/a")
        ax.relim(); ax.autoscale_view()
        ax.set_title(f"Fixes: {len(lats)}  Lat: {lat:.6f}  Lon: {lon:.6f}  Curv: {curv_txt}")

    return track_line, curr_pt


# --------------------------------------------------------------------------- #
#  Kick‑off
# --------------------------------------------------------------------------- #
ani = FuncAnimation(fig,
                    update,
                    interval=int(1000 / args.hz),
                    cache_frame_data=False)

try:
    plt.show()
finally:
    csv_file.close()
    print(f"CSV closed with {len(lats)} total fixes.")
