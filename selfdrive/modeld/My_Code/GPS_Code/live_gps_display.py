#!/usr/bin/env python3
"""
gps_display_centered_model_kalman.py
====================================

Vehicle-centred live viewer with:
  • reference-route match & look-ahead preview
  • heading-up display with arrow
  • modelV2.position “ghost” path
  • fixed square window around the car (–-window)

Changes vs. the original
------------------------
* Uses `liveLocationKalmanDEPRECATED` (Kalman-fused fix) instead of `gpsLocation`.
* Lat/Lon/Alt come from `positionGeodetic.value`.
* Bearing/speed derived from `velocityNED.value`.
* SubMaster topic list updated accordingly.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cereal.messaging as messaging

# ────────── CLI ────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--route",  type=Path,  default="/home/adas/openpilot/kalman_track_2025-06-12_10-50-29.csv",
                help="reference path (.csv / .rlog / .bz2)")
ap.add_argument("--thresh", type=float, default=20,  help="match threshold (m)")
ap.add_argument("--n",      type=int,   default=15,  help="# future points to plot")
ap.add_argument("--hz",     type=float, default=100, help="UI refresh rate (Hz)")
ap.add_argument("--window", type=float, default=20,  help="half window size (m)")
args = ap.parse_args()

# ────────── geo helpers ────────────────────────────────────────────────────
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
    """Return ground-speed (m/s) and bearing (rad) from N/E components."""
    vn, ve, *_ = v_ned
    spd = float(np.hypot(vn, ve))
    brg = float((np.arctan2(ve, vn) + 2*np.pi) % (2*np.pi))
    return spd, brg

# ────────── load reference route ───────────────────────────────────────────
route_lats, route_lons = [], []
if args.route.suffix == ".csv":
    with args.route.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            lat = row.get("lat") or row.get("latitude")
            lon = row.get("lon") or row.get("longitude")
            if lat and lon:
                route_lats.append(float(lat)); route_lons.append(float(lon))
else:
    try:
        from tools.lib.logreader import LogReader
    except ImportError:
        sys.exit("Need tools.lib.logreader for rlog preload")
    for ev in LogReader(str(args.route)):
        if ev.which() != "liveLocationKalmanDEPRECATED":             # accepts Kalman logs
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

# ────────── messaging ──────────────────────────────────────────────────────
sm = messaging.SubMaster(["liveLocationKalmanDEPRECATED", "modelV2"])
live_lats, live_lons = [], []

# ────────── Matplotlib set-up ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
fig.canvas.manager.set_window_title("Vehicle-centred Kalman GPS + model path")

route_line, = ax.plot([], [], ':',  color="grey", alpha=.5, label="reference")
pred_line,  = ax.plot([], [], lw=3,  color="tab:blue",   label=f"next {args.n}")
model_line, = ax.plot([], [], '--', color="tab:purple", label="modelV2 path")
curr_pt,    = ax.plot([], [], "ro", ms=6, label="current")
heading_q   = ax.quiver([], [], [], [],
                        angles='xy', scale_units='xy', scale=1,
                        color='limegreen', width=0.012, label="heading")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_aspect("equal", adjustable="box"); ax.grid(True); ax.legend(loc="upper right")

W = args.window   # half-width/height of view

# ────────── animation loop ─────────────────────────────────────────────────
def update(_):
    sm.update(0)
    if not sm.updated["liveLocationKalmanDEPRECATED"]:
        return
    g = sm["liveLocationKalmanDEPRECATED"]

    # validity gate
    if g.status != "valid" or not g.positionGeodetic.valid:
        return

    lat, lon, _ = g.positionGeodetic.value
    live_lats.append(lat); live_lons.append(lon)

    # bearing – prefer velocity; fall back to last-step diff
    if g.velocityNED.valid:
        _, bearing = horiz_speed_bearing(g.velocityNED.value)
    elif len(live_lats) > 1:
        _gx, _gy = equirect(np.array(live_lats[-2:]), np.array(live_lons[-2:]),
                            origin_lat, origin_lon)
        dx, dy = _gx[-1]-_gx[-2], _gy[-1]-_gy[-2]
        bearing = np.arctan2(dx, dy) if (dx or dy) else 0.0
    else:
        bearing = 0.0

    # ----- look-ahead points on reference route ----------------------------
    dists = haversine(route_lats, route_lons, lat, lon)
    idx = int(np.argmin(dists))
    nxt_lat = route_lats[idx:idx+args.n] if dists[idx] < args.thresh else np.array([])
    nxt_lon = route_lons[idx:idx+args.n] if dists[idx] < args.thresh else np.array([])

    # ----- project to planar coords, translate so car is origin ------------
    gx,  gy  = equirect(np.asarray(live_lats), np.asarray(live_lons),
                        origin_lat, origin_lon)
    grx, gry = equirect(route_lats, route_lons, origin_lat, origin_lon)
    gxp, gyp = equirect(nxt_lat, nxt_lon, origin_lat, origin_lon)

    tx, ty = gx[-1], gy[-1]       # translation
    gx  -= tx; gy  -= ty
    grx -= tx; gry -= ty
    gxp -= tx; gyp -= ty

    # rotate to heading-up
    x,  y  = rotate(gx,  gy,  -bearing)
    rx, ry = rotate(grx, gry, -bearing)
    xp, yp = rotate(gxp, gyp, -bearing) if len(gxp) else ([], [])

    # ----- modelV2 path (already car-centric, X fwd, Y left) --------------
    if sm.updated["modelV2"]:
        m = sm["modelV2"]
        if len(m.position.x):
            fwd  = np.array(m.position.x)        # forward (X) ➜ +Y after swap
            left = np.array(m.position.y)        # left (Y) ➜ +X after swap
            mdl_x, mdl_y = left, fwd
            model_line.set_data(mdl_x, mdl_y)

    # ----- update artists --------------------------------------------------
    route_line.set_data(rx, ry)
    pred_line.set_data(xp, yp)
    curr_pt.set_data([0], [0])
    heading_q.set_offsets([[0, 0]]); heading_q.set_UVC([0], [10])

    ax.set_xlim(-W, W); ax.set_ylim(-W, W)
    ax.set_title(f"Fixes {len(live_lats)} | "
                 f"Lat {lat:.5f} Lon {lon:.5f} | "
                 f"match {dists[idx]:.1f} m" if len(dists) else "")
    return

ani = FuncAnimation(fig, update,
                    interval=int(1000/args.hz),
                    cache_frame_data=False)
plt.show()
