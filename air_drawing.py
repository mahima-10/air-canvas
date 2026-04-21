"""
Air Drawing — touchless drawing using hand gestures via webcam.

Gestures (classified by MediaPipe's pretrained GestureRecognizer):
    Pointing_Up        -> DRAW
    Victory (peace)    -> SELECT (hover toolbar or thickness slider)

Toolbar tiles: colors, ERASE, EYEDROP, CLEAR.
Slider below toolbar: hover with SELECT to change brush size.

Keys:  q=quit, c=clear, s=save PNG, z=undo
"""

import datetime
import os
import time
import urllib.request
from collections import deque

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ---------- Configuration ----------

CAM_WIDTH, CAM_HEIGHT = 1280, 720
TOOLBAR_HEIGHT = 72
SLIDER_HEIGHT = 44
UI_HEIGHT = TOOLBAR_HEIGHT + SLIDER_HEIGHT

THICK_MIN, THICK_MAX = 2, 40
DEFAULT_THICKNESS = 6
ERASE_THICKNESS_MULT = 4
UNDO_LIMIT = 30

PALETTE = [
    ("BLUE",    (255,  90,  60)),
    ("GREEN",   ( 60, 200,  90)),
    ("RED",     ( 60,  60, 230)),
    ("YELLOW",  ( 60, 220, 240)),
    ("PURPLE",  (220, 100, 220)),
    ("WHITE",   (245, 245, 245)),
    ("ERASE",   ( 40,  40,  40)),
    ("EYEDROP", ( 80, 140, 200)),
    ("FILL",    (100,  60, 180)),
]
ACTION_LABELS = {"ERASE", "EYEDROP", "FILL"}


# ---------- MediaPipe Gesture Recognizer ----------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task")

# MediaPipe's built-in gesture labels that we care about.
GESTURE_MAP = {
    "Pointing_Up": "DRAW",
    "Victory":     "SELECT",
    "Open_Palm":   "PALM",   # hold to clear the canvas
}

PALM_CLEAR_HOLD_MS = 600

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

INDEX_TIP = 8


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading gesture recognizer model -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model ready.")


def draw_hand(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_EDGES:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 180, 255), -1)


# ---------- One Euro Filter ----------

class OneEuro:
    def __init__(self, freq=60.0, mincutoff=1.0, beta=0.02, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x
        dt = max(t - self._t_prev, 1e-3)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


# ---------- UI ----------

def build_toolbar(width):
    bar = np.zeros((TOOLBAR_HEIGHT, width, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    slot_w = width // len(PALETTE)
    slots = []
    for i, (label, color) in enumerate(PALETTE):
        x1, x2 = i * slot_w, (i + 1) * slot_w
        slots.append((x1, x2, label, color))
        fill = color if label not in ACTION_LABELS else (55, 55, 55)
        cv2.rectangle(bar, (x1 + 4, 6), (x2 - 4, TOOLBAR_HEIGHT - 6), fill, -1)
        text_color = (20, 20, 20) if label not in ACTION_LABELS | {"BLUE", "RED", "PURPLE"} else (240, 240, 240)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = x1 + (slot_w - tw) // 2
        ty = TOOLBAR_HEIGHT // 2 + th // 2
        cv2.putText(bar, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)
    return bar, slots


def slot_at(x, slots):
    for x1, x2, label, color in slots:
        if x1 <= x < x2:
            return label, color
    return None, None


def slider_x_to_thickness(x, w):
    frac = np.clip(x / max(w - 1, 1), 0.0, 1.0)
    return int(round(THICK_MIN + frac * (THICK_MAX - THICK_MIN)))


def draw_slider(canvas, y0, w, thickness, active_color, hover_x=None):
    cv2.rectangle(canvas, (0, y0), (w, y0 + SLIDER_HEIGHT), (22, 22, 22), -1)
    track_y = y0 + SLIDER_HEIGHT // 2
    cv2.line(canvas, (12, track_y), (w - 12, track_y), (90, 90, 90), 2)
    cv2.circle(canvas, (24, track_y), THICK_MIN // 2 + 2, (180, 180, 180), -1)
    cv2.circle(canvas, (w - 24, track_y), THICK_MAX // 2, (180, 180, 180), -1)
    frac = (thickness - THICK_MIN) / max(THICK_MAX - THICK_MIN, 1)
    kx = int(12 + frac * (w - 24))
    cv2.circle(canvas, (kx, track_y), max(thickness // 2, 4), active_color, -1)
    cv2.circle(canvas, (kx, track_y), max(thickness // 2, 4) + 2, (255, 255, 255), 2)
    if hover_x is not None:
        cv2.line(canvas, (hover_x, y0 + 4), (hover_x, y0 + SLIDER_HEIGHT - 4), (0, 255, 255), 1)
    cv2.putText(canvas, f"SIZE {thickness}", (w - 130, y0 + SLIDER_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


# ---------- Main ----------

def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read from webcam.")
    h, w = frame.shape[:2]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    toolbar, slots = build_toolbar(w)

    active_color = PALETTE[0][1]
    active_label = PALETTE[0][0]
    thickness = DEFAULT_THICKNESS
    mode = "IDLE"
    prev_point = None
    undo_stack = deque(maxlen=UNDO_LIMIT)
    stroke_active = False
    eyedrop_armed = False

    fx = OneEuro(mincutoff=2.5, beta=0.05)
    fy = OneEuro(mincutoff=2.5, beta=0.05)

    palm_hold_start = None
    palm_fired = False
    palm_progress = 0.0

    save_flash_until = 0.0
    save_flash_text = ""

    options = mp_vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        # Keep tracking a partially-clipped hand; default 0.5 drops too fast
        # when fingers/wrist leave the frame.
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    recognizer = mp_vision.GestureRecognizer.create_from_options(options)
    last_ts = 0

    def push_undo():
        undo_stack.append(canvas.copy())

    def begin_stroke():
        nonlocal stroke_active
        if not stroke_active:
            push_undo()
            stroke_active = True

    def end_stroke():
        nonlocal stroke_active, prev_point
        stroke_active = False
        prev_point = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = max(int(time.monotonic() * 1000), last_ts + 1)
        last_ts = ts_ms
        try:
            result = recognizer.recognize_for_video(mp_image, ts_ms)
        except Exception as e:
            print(f"[recognize skip] {e}")
            result = None

        # Keep a pristine copy of the webcam frame for eyedropper sampling — we
        # don't want to sample a pixel that already has the hand skeleton
        # overlay drawn on it.
        clean_frame = frame.copy()

        cursor = None
        sample_point = None          # where the eyedropper would read
        gesture = "IDLE"
        top_gesture_label = ""

        if result is not None and result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            ix = landmarks[INDEX_TIP].x * w
            iy = landmarks[INDEX_TIP].y * h
            now = time.monotonic()
            sx = int(fx(ix, now))
            sy = int(fy(iy, now))
            cursor = (sx, sy)

            # Compute an eyedropper sample point ~60 px beyond the fingertip,
            # along the direction from index PIP (6) to tip (8). That's where
            # you're "pointing" — not where your finger is blocking the view.
            px = landmarks[6].x * w
            py = landmarks[6].y * h
            dx, dy = ix - px, iy - py
            norm = (dx * dx + dy * dy) ** 0.5 + 1e-6
            ox = int(ix + (dx / norm) * 60)
            oy = int(iy + (dy / norm) * 60)
            if 0 <= ox < w and 0 <= oy < h:
                sample_point = (ox, oy)

            if result.gestures and result.gestures[0]:
                top = result.gestures[0][0]
                top_gesture_label = f"{top.category_name}({top.score:.2f})"
                gesture = GESTURE_MAP.get(top.category_name, "IDLE")

            # Fallback: Google's classifier only fires on textbook poses.
            # Accept casual pointing poses via two cheap landmark checks:
            #   (a) index tip clearly higher than middle/ring/pinky tips
            #       (works when hand is roughly upright)
            #   (b) index tip farther from the wrist in 3D than any other
            #       finger tip (works when the hand points toward the camera
            #       or sideways, where tips share a y-row but index sticks out)
            if gesture == "IDLE":
                def _d3(a, b):
                    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
                wrist = landmarks[0]
                iy_n = landmarks[8].y
                other_tips_y = [landmarks[i].y for i in (12, 16, 20)]
                idx_reach = _d3(landmarks[8], wrist)
                other_reaches = [_d3(landmarks[i], wrist) for i in (12, 16, 20)]
                if (iy_n < min(other_tips_y) - 0.04
                        or idx_reach > max(other_reaches) * 1.15):
                    gesture = "DRAW"
                    if not top_gesture_label:
                        top_gesture_label = "Pointing(fallback)"

            draw_hand(frame, landmarks, w, h)
        else:
            fx.reset()
            fy.reset()

        # ----- Act on gesture -----
        now = time.monotonic()

        # Palm-to-clear: hold Open_Palm for PALM_CLEAR_HOLD_MS to wipe the canvas.
        if gesture == "PALM":
            if palm_hold_start is None:
                palm_hold_start = now
                palm_fired = False
            palm_progress = min((now - palm_hold_start) / (PALM_CLEAR_HOLD_MS / 1000.0), 1.0)
            if palm_progress >= 1.0 and not palm_fired:
                push_undo()
                canvas[:] = 0
                palm_fired = True
        else:
            palm_hold_start = None
            palm_fired = False
            palm_progress = 0.0

        if gesture == "SELECT" and cursor:
            cy = cursor[1]
            if cy < TOOLBAR_HEIGHT:
                label, color = slot_at(cursor[0], slots)
                if label == "ERASE":
                    active_label = "ERASE"
                    eyedrop_armed = False
                elif label == "EYEDROP":
                    active_label = "EYEDROP"
                    eyedrop_armed = True
                elif label == "FILL":
                    # Don't overwrite active_color — fill uses the last picked color.
                    active_label = "FILL"
                    eyedrop_armed = False
                elif label:
                    active_color = color
                    active_label = label
                    eyedrop_armed = False
            elif cy < UI_HEIGHT:
                thickness = slider_x_to_thickness(cursor[0], w)
            mode = "SELECT"
            end_stroke()

        elif gesture == "PALM":
            mode = "PALM"
            end_stroke()

        elif gesture == "DRAW" and cursor and cursor[1] > UI_HEIGHT:
            if active_label == "EYEDROP" and eyedrop_armed:
                mode = "EYEDROP"
                if sample_point is not None:
                    bgr = clean_frame[sample_point[1], sample_point[0]]
                    active_color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
                    active_label = "CUSTOM"
                    eyedrop_armed = False
                prev_point = None
            elif active_label == "FILL":
                mode = "FILL"
                # Fire the flood fill only once per "tap" — when the DRAW
                # gesture first begins. Subsequent frames in the same gesture
                # don't re-fill. Lifting (IDLE/SELECT) resets stroke_active.
                if not stroke_active:
                    push_undo()
                    stroke_active = True
                    h_, w_ = canvas.shape[:2]
                    sx_ = max(0, min(w_ - 1, int(cursor[0])))
                    sy_ = max(0, min(h_ - 1, int(cursor[1])))
                    ff_mask = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
                    cv2.floodFill(canvas, ff_mask, (sx_, sy_), active_color,
                                  (0, 0, 0), (0, 0, 0), cv2.FLOODFILL_FIXED_RANGE)
                prev_point = None
            elif active_label == "ERASE":
                mode = "ERASE"
                begin_stroke()
                if prev_point is not None:
                    cv2.line(canvas, prev_point, cursor, (0, 0, 0),
                             max(thickness * ERASE_THICKNESS_MULT, 8))
                prev_point = cursor
            else:
                mode = "DRAW"
                begin_stroke()
                if prev_point is not None:
                    cv2.line(canvas, prev_point, cursor, active_color, thickness)
                prev_point = cursor

        else:
            mode = "IDLE"
            end_stroke()

        # ----- Composite output -----
        mask = canvas.any(axis=2)
        output = frame.copy()
        output[mask] = canvas[mask]

        output[:TOOLBAR_HEIGHT] = toolbar
        for x1, x2, label, _ in slots:
            if label == active_label or (active_label == "CUSTOM" and label == "EYEDROP" and eyedrop_armed):
                cv2.rectangle(output, (x1 + 2, 2), (x2 - 2, TOOLBAR_HEIGHT - 2), (0, 255, 255), 3)
                break

        slider_hover = cursor[0] if (cursor and gesture == "SELECT"
                                     and TOOLBAR_HEIGHT <= cursor[1] < UI_HEIGHT) else None
        draw_slider(output, TOOLBAR_HEIGHT, w, thickness, active_color, slider_hover)

        if cursor:
            ring_color = active_color if active_label != "ERASE" else (200, 200, 200)
            r = max(thickness // 2, 6)
            cv2.circle(output, cursor, r, ring_color, 2)
            cv2.circle(output, cursor, 3, ring_color, -1)

        # Eyedropper preview: show a crosshair + color swatch at the sample
        # point while armed, so the user can aim at the color they want.
        if eyedrop_armed and sample_point is not None:
            sx_, sy_ = sample_point
            bgr = clean_frame[sy_, sx_]
            swatch = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            cv2.line(output, (sx_ - 12, sy_), (sx_ + 12, sy_), (0, 0, 0), 2)
            cv2.line(output, (sx_, sy_ - 12), (sx_, sy_ + 12), (0, 0, 0), 2)
            cv2.line(output, (sx_ - 12, sy_), (sx_ + 12, sy_), (255, 255, 255), 1)
            cv2.line(output, (sx_, sy_ - 12), (sx_, sy_ + 12), (255, 255, 255), 1)
            cv2.circle(output, (sx_, sy_), 18, swatch, -1)
            cv2.circle(output, (sx_, sy_), 18, (255, 255, 255), 2)

        # Palm-clear hold progress ring around the cursor.
        if gesture == "PALM" and cursor and palm_progress > 0:
            radius = 40
            start_angle = -90
            end_angle = int(start_angle + 360 * palm_progress)
            cv2.ellipse(output, cursor, (radius, radius), 0,
                        start_angle, end_angle, (0, 255, 255), 4)
            label = "CLEARED" if palm_fired else "CLEAR…"
            cv2.putText(output, label, (cursor[0] - 40, cursor[1] + radius + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        tool_display = active_label if active_label != "CUSTOM" else "CUSTOM(eyedrop)"
        hud = (f"Mode:{mode}  Tool:{tool_display}  Size:{thickness}  "
               f"Undo:{len(undo_stack)}  Gesture:{top_gesture_label}  "
               f"|  q quit  c clear  s save  z undo")
        cv2.putText(output, hud, (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)

        if time.monotonic() < save_flash_until and save_flash_text:
            color = (0, 220, 0) if save_flash_text.startswith("Saved") else (0, 0, 220)
            (tw, th), _ = cv2.getTextSize(save_flash_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(output, (16, UI_HEIGHT + 12),
                          (28 + tw, UI_HEIGHT + 24 + th), (0, 0, 0), -1)
            cv2.putText(output, save_flash_text, (22, UI_HEIGHT + 18 + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        cv2.imshow("Air Drawing", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            push_undo()
            canvas[:] = 0
        elif key == ord('s'):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SAVE_DIR, f"air_drawing_{ts}.png")
            ok_save = cv2.imwrite(path, output)
            if ok_save:
                save_flash_until = time.monotonic() + 1.8
                save_flash_text = f"Saved: {os.path.basename(path)}"
                print(save_flash_text, "->", path)
            else:
                save_flash_until = time.monotonic() + 1.8
                save_flash_text = "Save FAILED"
                print("Save failed writing to", path)
        elif key == ord('z'):
            if undo_stack:
                canvas[:] = undo_stack.pop()
                end_stroke()

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()


if __name__ == "__main__":
    main()
