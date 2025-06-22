#!/usr/bin/env python3
"""
gps_display.py  •  live GPS plotting + on-disk CSV logging
=========================================================

  ── Live red dot & blue path           (Matplotlib)
  ── Optional historical preload log    (--preload FILE.rlog.bz2)
  ── Adjustable refresh rate            (--hz 10)
  ── Continuous CSV logging             (--out track.csv)

Example
-------
  # Just watch + save
  python gps_display.py

  # Preload an rlog, plot at 5 Hz, save to a named CSV
  python gps_display.py --preload 2024-05-12.rlog.bz2 --hz 5 --out may12_track.csv
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cereal.messaging as messaging

try:                                 # historical reader is optional
    from tools.lib.logreader import LogReader
except ImportError:
    LogReader = None


# --------------------------------------------------------------------------- #
#  Args
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--hz", type=float, default=100,
                   help="animation refresh rate (Hz)")
    p.add_argument("--preload", type=Path,
                   help="rlog / bz2 file to draw before going live")
    p.add_argument("--out", type=Path,
                   help="CSV path for storing fixes. "
                        "If omitted, an auto-named file is created.")
    p.add_argument("--name", type=Path, default="6-11",
                   help="CSV path for storing fixes. "
                        "If omitted, an auto-named file is created.")
    return p

args = build_parser().parse_args()


# --------------------------------------------------------------------------- #
#  Output CSV setup
# --------------------------------------------------------------------------- #
if args.out:
    csv_path = args.out.expanduser()
else:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = Path(f"gps_track_{args.name}.csv")

csv_path.parent.mkdir(parents=True, exist_ok=True)
csv_file = csv_path.open("w", newline="", buffering=1)  # line-buffered
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["unix_ms", "latitude", "longitude",
                     "altitude_m", "speed_mps", "bearing_deg"])
print(f"Logging GPS fixes to {csv_path.resolve()}")

# --------------------------------------------------------------------------- #
#  Historical preload
# --------------------------------------------------------------------------- #
lats, lons = [], []

if args.preload and args.preload.exists():
    if LogReader is None:
        print("Preload requested but tools.lib.logreader not available; skipping.")
    else:
        print(f"Pre-loading {args.preload} …")
        for ev in LogReader(str(args.preload)):
            if ev.which() != "gpsLocation":
                continue
            g = ev.gpsLocation
            valid = (getattr(g, "hasFix", True)
                     if g.which() == "gpsLocation" else (g.flags & 1))
            if not valid:
                continue
            lats.append(g.latitude)
            lons.append(g.longitude)
        print(f"  Loaded {len(lats)} points.")


# --------------------------------------------------------------------------- #
#  Projection helper
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
sm = messaging.SubMaster(["liveLocationKalmanDEPRECATED"])
origin_latlon = [None, None]

# --------------------------------------------------------------------------- #
#  Matplotlib
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6, 6))
fig.canvas.manager.set_window_title("GPS track (live)")

track_line, = ax.plot([], [], lw=2, color="tab:blue", label="path history")
curr_pt,   = ax.plot([], [], "ro", ms=6, label="current fix")

ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.legend(loc="upper left")


# --------------------------------------------------------------------------- #
#  Animation callback
# --------------------------------------------------------------------------- #
def update(_):
    sm.update(0)                     # non-blocking poll

    if sm.updated["liveLocationKalmanDEPRECATED"]:
        g = sm["liveLocationKalmanDEPRECATED"]

        # validity check (hasFix on new schema, bit-0 of flags on old)
        if hasattr(g, "hasFix"):
            if not g.hasFix:
                return track_line, curr_pt
        elif hasattr(g, "flags") and (g.flags & 1) == 0:
            return track_line, curr_pt

        lat, lon = g.latitude, g.longitude
        if origin_latlon[0] is None:
            origin_latlon[:] = [lat, lon]

        lats.append(lat); lons.append(lon)

        # --- write to CSV ----------------------------------------------------
        csv_writer.writerow([g.unixTimestampMillis,
                             lat, lon, g.altitude,
                             g.speed, g.bearingDeg])
        if len(lats) % 100 == 0:
            print(f"  stored {len(lats)} fixes…")

        # --- update plot ------------------------------------------------------
        x, y = equirect_xy(np.array(lats), np.array(lons),
                           origin_latlon[0], origin_latlon[1])

        track_line.set_data(x, y)
        curr_pt.set_data(x[-1:], y[-1:])

        ax.relim(); ax.autoscale_view()
        ax.set_title(f"Fixes: {len(lats)}  "
                     f"Lat: {lat:.6f}  Lon: {lon:.6f}")

    return track_line, curr_pt


# --------------------------------------------------------------------------- #
#  Kick-off
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
