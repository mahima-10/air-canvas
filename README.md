# Air Drawing

Draw in mid-air with your hand using a webcam. Hand gestures pick colors, change brush size, fill shapes, erase, and clear the canvas — no keyboard or mouse needed.

Built with OpenCV + MediaPipe (Gesture Recognizer + Hand Landmarker). Runs locally, CPU-only.

## Features

- Real-time fingertip tracking with One Euro Filter smoothing
- Gesture recognition via Google's pretrained MediaPipe model, with landmark-based fallbacks for casual poses
- Color palette, ERASE, EYEDROP (sample pixel color from the webcam), FILL (flood fill)
- Thickness slider (hover to set)
- Palm-to-clear gesture
- Undo stack
- Save to PNG

## Install

Requires Python 3.11 (MediaPipe doesn't ship `solutions` or stable wheels for 3.13 on macOS).

```bash
cd air-drawing
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads `gesture_recognizer.task` (~8 MB) next to the script. It's cached after that.

## Run

```bash
python air_drawing.py
```

macOS will ask for camera permission the first time.

## Gestures

| Pose | Action |
| --- | --- |
| Index finger up (`Pointing_Up`) | Draw / tap tool |
| Peace sign (`Victory`) | Hover toolbar or thickness slider to select |
| Open palm held ~0.6 s | Clear canvas |

Tools on the toolbar (left to right): **BLUE**, **GREEN**, **RED**, **YELLOW**, **PURPLE**, **WHITE**, **ERASE**, **EYEDROP**, **FILL**.

- **EYEDROP**: select it, then point at any pixel in the webcam view. The color under a crosshair ~60 px beyond your fingertip is sampled. Aim with the crosshair preview.
- **FILL**: select a color first, then select FILL, then tap inside a closed outline. Repeats for each tap until you pick another tool.

## Keys

| Key | Action |
| --- | --- |
| `q` | Quit |
| `c` | Clear canvas |
| `z` | Undo |
| `s` | Save PNG (`air_drawing_<timestamp>.png` next to the script) |

## Tips

- Keep your whole hand in frame. Roughly arm's length from the camera works best — closer than that and the gesture classifier struggles.
- Good lighting and a plain background help a lot.
- If the classifier doesn't recognize your pointing pose, the fallback ("Pointing(fallback)" in the HUD) should still trigger DRAW as long as your index finger is clearly extended past the other fingers.
