import cv2
import numpy as np


def draw_cursor(canvas: np.ndarray, x: int, y: int, visible: bool = True):
    if not visible:
        return

    h, w = canvas.shape[:2]

    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    tip = (x, y)
    left = (x - 5, y + 12)
    down = (x, y + 10)
    right = (x + 5, y + 12)

    pts = np.array([tip, left, down, right], dtype=int)
    cv2.fillPoly(canvas, [pts], (240, 240, 240), cv2.LINE_AA)
    cv2.polylines(canvas, [pts], False, (40, 40, 40), 1, cv2.LINE_AA)

    cv2.line(canvas, tip, down, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(canvas, down, left, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(canvas, down, right, (40, 40, 40), 1, cv2.LINE_AA)