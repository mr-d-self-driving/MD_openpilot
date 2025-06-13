#!/usr/bin/env python3
import cereal.messaging as messaging
import numpy as np
import matplotlib.pyplot as plt
from cereal import log
from cereal.messaging import PubMaster, SubMaster


def run_realtime_plot():
    # sub = messaging.sub_sock('modelV2', conflate=True)
    # sub_cs = messaging.sub_sock('carState', conflate=True)
    # sub_nav = messaging.sub_sock('navInstruction', conflate=True)
    sm = SubMaster(['modelV2'])

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(-30,30)
    ax.set_ylim(0,80)
    ax.set_title("FrogPilot Lane-Lines & Road-Edges (modelV2)")
    ax.grid(True)

    lane_lines = []
    planed_path = []
    road_edges = []
    veh_dot = ax.plot(0, 0, 'ro', markersize=8, label="Vehicle")[0]
    first_frame = True

    while True:
        sm.update(0)
        m = sm['modelV2']

        if first_frame:
            for artist in lane_lines + road_edges:
                artist.remove()
            lane_lines.clear()
            planed_path.clear()
            road_edges.clear()


            pp_line, = ax.plot([], [], lw=5, label=f"PlanedPath")
            planed_path.append(pp_line)
            for i in range(4):
                line, = ax.plot([], [], lw=2, label=f"Lane {i}")
                lane_lines.append(line)

            for i in range(len(m.roadEdges)):
                edge, = ax.plot([], [], lw=1.5, linestyle='--', label=f"Edge {i}")
                road_edges.append(edge)

            ax.legend(loc="lower left")
            first_frame = False

        pp = m.position
        xs = np.array(pp.x); ys = np.array(pp.y)
        mask = ~np.isnan(xs) & ~np.isnan(ys)
        planed_path[0].set_data(ys[mask], xs[mask])


        veh_dot.set_xdata([0])
        veh_dot.set_ydata([0])

        # autoscale + draw
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()

        fig.canvas.flush_events()

if __name__ == "__main__":
    run_realtime_plot()
