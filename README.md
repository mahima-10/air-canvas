# Air Canvas

Draw in mid-air with your hand — no mouse, no keyboard, no touchscreen. A webcam watches your fingers; point to draw, peace-sign to pick colors/tools, open-palm to wipe the canvas.

Built with **OpenCV** and **MediaPipe's Gesture Recognizer** (with a landmark fallback for casual poses). Runs locally on CPU.

---

## Features

- Real-time fingertip tracking with One Euro Filter smoothing
- Gesture recognition via Google's pretrained MediaPipe model, with a landmark-based fallback when it doesn't fire
- **9 tools**: 6 colors, ERASE, EYEDROP (sample color from the webcam), FILL (flood fill), plus shape tools: LINE, RECT, CIRCLE
- Live shape preview while dragging — what you see is what gets committed
- Thickness slider (2–40 px)
- Palm-to-clear gesture with a radial hold indicator
- Undo stack (30 steps)
- Save to timestamped PNG

## Quick start

Requires **Python 3.11** (MediaPipe doesn't ship clean wheels for 3.13 on macOS).

```bash
git clone https://github.com/mahima-10/air-canvas.git
cd air-canvas
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python air_drawing.py
```

First run downloads `gesture_recognizer.task` (~8 MB) into the folder. macOS will ask for camera permission.

## Gestures

| Pose | Action |
| --- | --- |
| **Index finger up** (`Pointing_Up`) | Draw / tap |
| **Peace sign** (`Victory`) | Hover over toolbar or thickness slider to select |
| **Open palm** held ~0.6 s (`Open_Palm`) | Clear the canvas |

If the classifier doesn't recognize your pointing pose, a landmark fallback checks whether your index finger is genuinely extended past the others. Works for upright *or* horizontal "fist-with-pointer" poses.

## Tools

Peace-sign hover over the top toolbar to pick one:

- **BLUE, GREEN, RED, YELLOW, PURPLE, WHITE** — pen colors
- **ERASE** — paint the canvas background (4× brush thickness)
- **EYEDROP** — samples a pixel ~60 px beyond your fingertip, in the direction you're pointing. A crosshair + color swatch shows the live sample while armed.
- **FILL** — pick a color first, then FILL, then point inside a closed outline to bucket-fill
- **LINE / RECT / CIRCLE** — first DRAW frame sets the anchor (yellow dot), move your hand for a live preview, release (stop pointing) to commit

## Keys

| Key | Action |
| --- | --- |
| `q` | Quit |
| `c` | Clear canvas |
| `z` | Undo |
| `s` | Save PNG as `air_drawing_<timestamp>.png` |

## Tips

- Keep your **whole hand in frame**. Arm's-length distance from the camera works best — too close and the classifier struggles with partial hands.
- Bright, even lighting and a plain background make tracking more reliable.
- If drawing feels flaky, watch the `Gesture:` field in the bottom HUD — it shows the raw classification and confidence, which makes it easy to see why a gesture isn't firing.

## How it works

1. Webcam frame is captured, mirrored, converted to RGB.
2. MediaPipe's `GestureRecognizer` runs in VIDEO mode, returning 21 hand landmarks + a top-1 gesture label.
3. Index fingertip (landmark 8) is smoothed with the One Euro Filter and used as the cursor.
4. If the classifier's label is `Pointing_Up` / `Victory` / `Open_Palm`, it maps to DRAW / SELECT / PALM. Otherwise a cheap landmark check decides whether to force DRAW anyway.
5. The canvas is a NumPy array composited over the webcam frame. Strokes / shapes / fills go into the canvas; the toolbar, slider, HUD, and hand skeleton are drawn on top.
