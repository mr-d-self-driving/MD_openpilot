#!/usr/bin/env python3
"""
gps_display_centered_model.py
=============================

Live vehicle-centred GPS viewer with:
  • reference route match & look-ahead
  • vehicle heading arrow
  • **modelV2.position** preview         <-- NEW
  • fixed square window around the car
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cereal.messaging as messaging

# ────────── CLI ────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--route",  type=Path,  default="/home/adas/openpilot/gps_track_6-11.csv",
               help="reference path (.csv / .rlog / .bz2)")
p.add_argument("--thresh", type=float, default=20,  help="match threshold (m)")
p.add_argument("--n",      type=int,   default=15,   help="# future points to plot")
p.add_argument("--hz",     type=float, default=100,  help="UI refresh rate (Hz)")
p.add_argument("--window", type=float, default=50,  help="half window size (m)")
args = p.parse_args()

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
        if ev.which() != "gpsLocation": continue
        g = ev.gpsLocation
        if getattr(g, "hasFix", True) or (getattr(g, "flags", 1) & 1):
            route_lats.append(g.latitude); route_lons.append(g.longitude)

if not route_lats:
    sys.exit("No GPS points in route file.")
route_lats, route_lons = map(np.asarray, (route_lats, route_lons))
origin_lat, origin_lon = route_lats[0], route_lons[0]
print(f"Loaded {len(route_lats)} route points from {args.route}")

# ────────── messaging ──────────────────────────────────────────────────────
sm = messaging.SubMaster(["gpsLocation", "modelV2"])
live_lats, live_lons = [], []

# ────────── Matplotlib set-up ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,6))
fig.canvas.manager.set_window_title("Vehicle-centred GPS + model path")

hist_line,  = ax.plot([], [], lw=2,  color="tab:green",  label="history")
route_line, = ax.plot([], [], ':',  color="grey", alpha=.6, label="reference")
pred_line,  = ax.plot([], [], lw=3,  color="tab:blue",
                      label=f"next {args.n} from route")
model_line, = ax.plot([], [], '--', color="tab:purple", label="modelV2 path")  # NEW
curr_pt,    = ax.plot([], [], "ro", ms=6, label="current")
heading_q   = ax.quiver([], [], [], [],
                        angles='xy', scale_units='xy', scale=1,
                        color='limegreen', width=0.012, label="heading")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_aspect("equal", adjustable="box"); ax.grid(True); ax.legend()

W = args.window   # half-width/height of view

# ────────── animation loop ─────────────────────────────────────────────────
def update(_):
    sm.update(0)

    # --- GPS ----------------------------------------------------------------
    # if not sm.updated["gpsLocation"]:
    #     return
    g = sm["gpsLocation"]
    if hasattr(g,"hasFix") and not g.hasFix:
        return
    lat, lon = g.latitude, g.longitude
    live_lats.append(lat); live_lons.append(lon)

    # derive bearing (rad)
    if getattr(g, "bearingDeg", 0) > 0:
        bearing = deg2rad(g.bearingDeg)
    elif len(live_lats) > 1:
        gx2, gy2 = equirect(np.array(live_lats[-2:]), np.array(live_lons[-2:]),
                            origin_lat, origin_lon)
        dx, dy = gx2[-1]-gx2[-2], gy2[-1]-gy2[-2]
        bearing = np.arctan2(dx, dy) if (dx or dy) else 0.0
    else:
        bearing = 0.0

    # --- future points from route ------------------------------------------
    dists = haversine(route_lats, route_lons, lat, lon)
    idx = int(np.argmin(dists))
    if dists[idx] < args.thresh:
        nxt_lat = route_lats[idx:idx+args.n]
        nxt_lon = route_lons[idx:idx+args.n]
    else:
        nxt_lat = nxt_lon = np.array([])

    # --- projection of global tracks ---------------------------------------
    gx,  gy  = equirect(np.asarray(live_lats), np.asarray(live_lons),
                        origin_lat, origin_lon)
    grx, gry = equirect(route_lats, route_lons, origin_lat, origin_lon)
    gxp, gyp = equirect(nxt_lat, nxt_lon, origin_lat, origin_lon)

    # translation so current fix is origin
    tx, ty = gx[-1], gy[-1]
    gx  -= tx;  gy  -= ty
    grx -= tx;  gry -= ty
    gxp -= tx;  gyp -= ty

    # rotation to heading-up
    x,  y  = rotate(gx,  gy,  -bearing)
    rx, ry = rotate(grx, gry, -bearing)
    xp, yp = rotate(gxp, gyp, -bearing) if len(gxp) else ([], [])

    # --- modelV2 path (already car-relative) -------------------------------
    if sm.updated["modelV2"]:
        m = sm["modelV2"]
        if len(m.position.x):             # arrays of same length
            fwd = np.array(m.position.x)
            left = np.array(m.position.y)
            mdl_x, mdl_y = left, fwd     # left→-X , forward→+Y
            model_line.set_data(mdl_x, mdl_y)
    # if no update this frame, line keeps previous data

    # --- update artists -----------------------------------------------------
    # hist_line.set_data(x, y)
    route_line.set_data(rx, ry)
    pred_line.set_data(xp, yp)
    curr_pt.set_data([0], [0])
    heading_q.set_offsets([[0,0]]); heading_q.set_UVC([0],[10])

    ax.set_xlim(-W, W); ax.set_ylim(-W, W)
    ax.set_title(f"Fixes {len(live_lats)} | "
                 f"Lat {lat:.5f} Lon {lon:.5f} | "
                 f"match {'%.1f m' % dists[idx] if dists[idx] < 1e4 else '∞'}")
    return

ani = FuncAnimation(fig, update,
                    interval=int(1000/args.hz),
                    cache_frame_data=False)
plt.show()
