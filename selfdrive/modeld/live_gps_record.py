#!/usr/bin/env python3
"""
gps_path_to_modelV2.py
======================

Live vehicle-centred viewer **and** real-time override of `modelV2.position`
with a GPS-derived path.

Key features
------------
* Uses Kalman-fused fixes (`liveLocationKalmanDEPRECATED`).
* Displays reference-route match, heading-up map, purple ghost of
  *whatever* we inject into `modelV2`.
* Publishes a fresh `modelV2` every animation frame, identical to the
  network output **except** for `position.{x,y,xStd,yStd}`, which now
  contain the resampled GPS path.
"""

from __future__ import annotations
import argparse, csv, sys, traceback
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cereal.messaging as messaging

# ─────────────── CLI ───────────────────────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--route",  type=Path, default="/home/adas/openpilot/kalman_track_2025-06-12_10-50-29.csv",
                help="reference path (.csv or .rlog.bz2)")
ap.add_argument("--thresh", type=float, default=20,  help="match threshold (m)")
ap.add_argument("--n",      type=int,   default=150,  help="# future points to plot")
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

# ───────── messaging ───────────────────────────────────────────────────────
sm = messaging.SubMaster(["liveLocationKalmanDEPRECATED", "modelV2"])
# pm = messaging.PubMaster(["modelV2"])

live_lats, live_lons = [], []

# ────────── Matplotlib boiler-plate ────────────────────────────────────────
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

# ─────────── animation callback ────────────────────────────────────────────
def update(_):
    try:
        sm.update(0)
        if not sm.updated["liveLocationKalmanDEPRECATED"]:
            return
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

        # ─── GPS path ➜ modelV2 override ───────────────────────────────────
        if sm.updated["modelV2"] and len(xp) >= 2:
            m_in  = sm["modelV2"]
            tgrid = np.array(m_in.position.t)     # 33-pt grid

            xp_r, yp_r = resample_xy_to_t(xp.astype(float),
                                          yp.astype(float),
                                          tgrid)

            # msg = messaging.new_message('modelV2')  # builder
            # msg.modelV2.CopyFrom(m_in)              # deep copy
            # msg.modelV2.position.x[:]    = yp_r.tolist()   # forward ➜ x
            # msg.modelV2.position.y[:]    = xp_r.tolist()   # left    ➜ y
            # msg.modelV2.position.xStd[:] = [0.05]*len(tgrid)
            # msg.modelV2.position.yStd[:] = [0.05]*len(tgrid)

            # pm.send('modelV2', msg)                 # publish
            model_line.set_data(xp_r, yp_r)         # debug ghost

        # ─── update plot ──────────────────────────────────────────────────
        route_line.set_data(rx, ry)
        pred_line .set_data(xp, yp)
        curr_pt   .set_data([0], [0])
        heading_q .set_offsets([[0, 0]]); heading_q.set_UVC([0], [10])

        ax.set_xlim(-W, W); ax.set_ylim(-W, W)
        ax.set_title(f"Fixes {len(live_lats)} | match {dists[idx]:.1f} m")
    except Exception:
        traceback.print_exc()

ani = FuncAnimation(fig, update,
                    interval=int(1000/args.hz),
                    cache_frame_data=False)
plt.show()
