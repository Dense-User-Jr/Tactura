# Gesture template replay

In Listen mode, the motion gate finds gyro bursts. Each burst can yield several 1.2-second clips centered on strong strokes. Calibration lines up the signed three-axis traces, averages them, and stores a few directions of variation as eigengestures. Matching tries a small time shift, then discounts limited variation along those directions. The best match can still be rejected for poor fit or insufficient strength. The live path also handles saved blueprint versions, negative corrections, and accelerometer-based shoves. The earlier frequency fingerprint was an experiment. Its magnitude spectrum can describe rhythm, but it loses the sign that helps distinguish opposite twists.

## Replay

Use NumPy and run from the repo root. The calibration files below are the first five recorded reps per gesture from November 10. The answer key is from November 15.

```sh
python3 ML/compare_gesture_templates.py \
  data/russell/calibration/00{01..25}_calib_*_20261110_*.csv \
  --labels data/russell/classify/labels_20261115.csv
```

The medoid is a real calibration rep with the smallest total distance to its peers. Comparing it with the mean under shifted RMS isolates the template choice. Comparing the medoid under shifted RMS and dynamic time warping isolates the alignment choice. Warping lets different parts of the clip shift by up to five samples, or 100 ms, with one alignment across all three signed axes. [WristRotate](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/8163) used DTW for wrist-worn motion, and [Kurbalija et al.](https://arxiv.org/abs/1107.0134) studied the effect of bounding the alignment.

| Method | Correct gestures | False activations |
| --- | ---: | ---: |
| Aligned mean with eigengestures | 137/201 | 5/41 |
| Aligned mean with shifted RMS | 133/201 | 5/41 |
| Recorded medoid with shifted RMS | 127/201 | 4/41 |
| Recorded medoid with bounded DTW | 137/201 | 8/41 |

All four missed all 55 Shakes with the first calibration batch. The reference called 50 of them Twist Left. As a sensitivity check, run the same command with `data/russell/calibration/*_20261110_*.csv` to pool eight additional Shake recordings from later that night. With those added reps, the reference and DTW both got 55/55 Shakes. Overall they scored 191/201 and 192/201, with 5/41 and 8/41 false activations, respectively. The replay points to the choice of Shake calibration reps as well as timing. It uses one participant and known development logs, with one template per class and windowed Classify input. Live Listen results and cross-user accuracy require fresh recordings.

## Wrist and phoneme trials

Try the current pose and a relaxed neutral pose on each wrist. For a candidate gesture definition, try a single deliberate outward or inward turn for each twist and several short alternating strokes for Shake. Record the watch wrist, starting pose, and actual motion, and ask which turns feel comfortable. Randomize Twist Left, Twist Right, Shake, idle, watch checks, and ordinary arm movement. Give each person separate calibration reps, then score the two-way Shake and Twist Left confusion and false activations by wrist and pose. Freeze the chosen definitions before collecting a fresh test round.

Listen already maps recognized gestures to two-sound bins. The spelling readout now shows both sounds for each of the last two gestures. Try spelling **this** with Flick Up (TH/DH), Twist Left (IH/UW), and Shake (S/Z). The edge of the screen shows the gesture map, and the center shows recent bins and word choices. Double Tap carries CH/JH in the mapping and still needs a usable calibration path before it can be trialed live.
