# Cap-confusion experiment (negative result)
Date: Oct 2026. Model: YOLO11s base (`weights/yolo11s_shwd_best.pt`).

## Problem
YouTube field test (`https://www.youtube.com/watch?v=PHezq3zCgGs`, 1940 frames):
cloth caps and bucket hats detected as `helmet` (e.g. grey cap @0.85).
Cause: only 2 classes exist (helmet/head); any head-covering defaults to helmet.

## What was tried
1. Mined 217 cap crops from the video (`scripts/cap_mining.py`, see `outputs/cap_grid.jpg`).
2. Added as background images (empty labels) to train set.
3. Fine-tuned 25 epochs from base (`runs/train/yolo11s_shwd_capfix`,
   weights: `weights/yolo11s_shwd_capfix_best.pt`).

## Result — fix FAILED
Same video, conf 0.5:

| Model   | helmet boxes | head boxes | val mAP50 |
|---------|--------------|------------|-----------|
| base    | 2093         | 987        | 0.948     |
| capfix  | 2480         | 551        | 0.940     |

Caps still fire as helmet (`outputs/capfix_frame1400.jpg` vs base frame).
One partial win: a bare-head false helmet was removed.

## Conclusion
Background-only fine-tuning cannot fix open-set confusion; it only shifts
calibration. Correct fix: 3rd `cap` class with annotated boxes (future work).
Base model retained as final.
