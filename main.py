import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

SLIDES = Path(__file__).parent / "slides"
STAGE, CAMERA = "Presentation", "camera"
COLORS = [(50, 50, 240), (80, 190, 60), (240, 120, 40)]
GESTURES = {
    (1, 1, 1, 1, 1): "start",
    (0, 1, 0, 0, 0): "next",
    (1, 0, 0, 0, 0): "prev",
    (0, 1, 1, 0, 0): "draw",
    (1, 1, 1, 0, 0): "color",
    (0, 1, 1, 1, 0): "erase",
}
HOLD = 1.0
SCALE, CAM_W = 1.25, 560
BRUSH, ERASER = 5, 26

mp_hands = mp.solutions.hands


def load(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)


def fingers(pts):
    d = lambda a, b: np.linalg.norm(pts[a] - pts[b])
    thumb = d(4, 5) > 0.7 * d(0, 9)
    return (thumb, *(d(t, 0) > d(t - 2, 0) for t in (8, 12, 16, 20)))


def draw_hand(img, pts):
    pts = pts.astype(int)
    for a, b in mp_hands.HAND_CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 3, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p), 4, (40, 40, 235), -1, cv2.LINE_AA)


def draw_status(img, tip, color, progress):
    if tip is not None:
        tip = tuple(tip.astype(int))
        cv2.circle(img, tip, 20, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.ellipse(img, tip, (20, 20), -90, 0, 360 * progress, color, 4, cv2.LINE_AA)
    y = img.shape[0] - 26
    for i, c in enumerate(COLORS):
        active = c == color
        cv2.circle(img, (28 + i * 40, y), 15 if active else 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (28 + i * 40, y), 13 if active else 6, c, -1, cv2.LINE_AA)


def main():
    slides = [load(p) for p in sorted(SLIDES.glob("*.png"))]
    h, w = slides[0].shape[:2]
    canvases = [np.zeros_like(s) for s in slides]
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    cv2.namedWindow(STAGE)
    cv2.namedWindow(CAMERA)
    cv2.moveWindow(STAGE, 20, 60)
    cv2.moveWindow(CAMERA, w + 40, 60)

    index, color = -1, 0
    held = pending = point = last = None
    since = changed = 0
    fired = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.medianBlur(cv2.flip(frame, 1), 3)
        fh, fw = frame.shape[:2]
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.time()

        cam = cv2.resize(frame, (CAM_W, CAM_W * fh // fw))
        gesture = tip = None
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            pts = np.array([(p.x * fw, p.y * fh) for p in lm])
            gesture = GESTURES.get(fingers(pts))
            target = np.clip((pts[8] / (fw, fh) - 0.2) / 0.6, 0, 1) * (w - 1, h - 1)
            point = target if point is None else point * 0.5 + target * 0.5
            pts *= CAM_W / fw
            tip = pts[8]
            draw_hand(cam, pts)
        else:
            point = None

        if gesture == held:
            pending = gesture
        elif gesture != pending:
            pending, changed = gesture, now
        elif now - changed > 0.2:
            held, since, fired, last = gesture, changed, False, None

        progress = min((now - since) / HOLD, 1) if held else 0
        ready = progress == 1
        if ready and not fired:
            fired = True
            if held == "start":
                index = max(index, 0)
                canvases[index][:] = 0
            elif held == "next" and index >= 0:
                index = min(index + 1, len(slides) - 1)
            elif held == "prev" and index >= 0:
                index = max(index - 1, 0)
            elif held == "color":
                color = (color + 1) % len(COLORS)

        stage = np.full((h, w, 3), 24, np.uint8)
        if index >= 0:
            canvas = canvases[index]
            cur = None if point is None else tuple(point.astype(int))
            if ready and cur and held in ("draw", "erase"):
                ink, size = (COLORS[color], BRUSH) if held == "draw" else ((0, 0, 0), ERASER * 2)
                cv2.line(canvas, last or cur, cur, ink, size)
                last = cur
            mask = canvas.any(axis=2)
            stage = slides[index].copy()
            stage[mask] = canvas[mask]
            if cur:
                radius = ERASER if held == "erase" else BRUSH + 3
                cv2.circle(stage, cur, radius, COLORS[color], 2, cv2.LINE_AA)

        draw_status(cam, tip, COLORS[color], progress)
        cv2.imshow(STAGE, stage)
        cv2.imshow(CAMERA, cam)

        key = cv2.waitKey(1) & 0xFF
        closed = min(cv2.getWindowProperty(n, cv2.WND_PROP_VISIBLE) for n in (STAGE, CAMERA)) < 1
        if key in (27, ord("q")) or closed:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
