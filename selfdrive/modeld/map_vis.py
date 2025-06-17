#!/usr/bin/env python3
"""
live_plot_nav.py  ––  real-time visualiser for nav_features (1×256)
                       and nav_instructions (1×150 → 50×3 grid)

Run on your development PC while replaying a route, or directly on
the Comma 3 / 3X (ssh -X).
"""
import numpy as np, matplotlib.pyplot as plt, cereal.messaging as messaging

sm = messaging.SubMaster(['navModelDEPRECATED', 'navInstruction'])

fig, (ax_feat, ax_instr) = plt.subplots(2, 1, figsize=(8, 7))
im = ax_instr.imshow(np.zeros((50, 3)), vmin=0, vmax=1, aspect='auto')
ax_feat.set_title('nav_features (256-D)')
ax_instr.set_title('nav_instructions  (dist bin ↑  ◄L  • S  • R►)')
nav_instructions = np.zeros(150, dtype=np.float32)
while True:
    sm.update()
    if sm.updated['navModelDEPRECATED']:
        nav_f = sm['navModelDEPRECATED'].features                    # shape (256,)  :contentReference[oaicite:0]{index=0}
        ax_feat.clear(); ax_feat.plot(nav_f); ax_feat.set_ylim([-2, 2])
    if sm.updated['navInstruction']:
        nav_instructions[:] = 0
        for maneuver in sm["navInstruction"].allManeuvers:
          distance_idx = 25 + int(maneuver.distance / 20)
          direction_idx = 0
          if maneuver.modifier in ("left", "slight left", "sharp left"):
            direction_idx = 1
          if maneuver.modifier in ("right", "slight right", "sharp right"):
            direction_idx = 2
          if 0 <= distance_idx < 50:
            nav_instructions[distance_idx*3 + direction_idx] = 1
    #     nav_i = sm['navInstruction'].value                 # shape (150,)
        im.set_data(nav_instructions.reshape(50, 3))                  # 50 distance buckets × {L,S,R}
    plt.pause(0.01)
