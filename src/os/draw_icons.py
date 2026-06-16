import cv2
import numpy as np


def draw_icon_shape(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                    shape: str, color: tuple):
    cx, cy = x + w // 2, y + h // 2

    if shape == "folder":
        pts = np.array([
            [x + 8, y + 4],
            [x + w - 8, y + 4],
            [x + w, y + 16],
            [x, y + 16],
        ], dtype=int)
        cv2.fillPoly(canvas, [pts], color)
        cv2.polylines(canvas, [pts], True, (max(0, color[0]-30), max(0, color[1]-30), max(0, color[2]-30)), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x + 4, y + 16), (x + w - 4, y + h - 4), color, -1, cv2.LINE_AA)
        cv2.line(canvas, (x + 4, y + 22), (x + w - 4, y + 22), (max(0, color[0]-30), max(0, color[1]-30), max(0, color[2]-30)), 1, cv2.LINE_AA)

    elif shape == "calc":
        cv2.rectangle(canvas, (x, y + 4), (x + w, y + h - 4), color, -1, cv2.LINE_AA)
        inner = (max(0, color[0]-40), max(0, color[1]-40), max(0, color[2]-40))
        cv2.rectangle(canvas, (x + 8, y + 12), (x + w - 8, y + h - 12), inner, -1, cv2.LINE_AA)
        for bx, by in [(x + 10, y + 14), (x + 22, y + 14), (x + 34, y + 14),
                       (x + 10, y + 26), (x + 22, y + 26), (x + 34, y + 26)]:
            cv2.rectangle(canvas, (bx, by), (bx + 10, by + 10), color, -1, cv2.LINE_AA)

    elif shape == "music":
        body_pts = np.array([
            [x + w // 2 + 6, y + 14],
            [x + w // 2 + 18, y + 14],
            [x + w // 2 + 22, y + h - 6],
            [x + w // 2 + 6, y + h - 6],
        ], dtype=int)
        cv2.fillPoly(canvas, [body_pts], color)
        cv2.ellipse(canvas, (x + w // 2 + 5, y + 18), (8, 6), 0, 0, 360, color, -1, cv2.LINE_AA)
        cv2.line(canvas, (x + w // 2 + 12, y + 12), (x + w // 2 + 12, y + h - 6), color, 2, cv2.LINE_AA)
        cv2.ellipse(canvas, (x + w // 2 + 5, y + 18), (8, 6), 20, 200, 340, color, 1, cv2.LINE_AA)

    elif shape == "globe":
        cv2.circle(canvas, (cx, cy + 2), 16, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy + 2), 16, (max(0, color[0]-40), max(0, color[1]-40), max(0, color[2]-40)), 1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy + 2), (16, 16), 0, 0, 180, color, 1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy + 2), (16, 16), 0, 180, 360, (max(0, color[0]-40), max(0, color[1]-40), max(0, color[2]-40)), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx - 16, cy + 2), (cx + 16, cy + 2), color, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - 14), (cx, cy + 18), color, 1, cv2.LINE_AA)

    elif shape == "camera":
        body_pts = np.array([[x + 10, y + 14], [x + w - 10, y + 14],
                              [x + w - 10, y + h - 8], [x + 10, y + h - 8]], dtype=int)
        cv2.fillPoly(canvas, [body_pts], color)
        cv2.circle(canvas, (cx, cy + 2), 11, (230, 230, 230), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy + 2), 7, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy + 2), 3, (230, 230, 230), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x + w // 2 - 3, y + 6), (x + w // 2 + 3, y + 14), color, -1, cv2.LINE_AA)

    elif shape == "gear":
        inner_r = 7
        outer_r = 16
        num_teeth = 8
        pts_out = []
        pts_in = []
        for i in range(num_teeth * 2):
            angle = np.pi * i / num_teeth
            r = outer_r if i % 2 == 0 else outer_r - 5
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))
            if i % 2 == 0:
                pts_out.append([px, py])
            else:
                pts_in.append([px, py])

        outer_pts = np.array(pts_out, dtype=int)
        cv2.fillPoly(canvas, [outer_pts], color)
        cv2.circle(canvas, (cx, cy), inner_r, (26, 26, 46), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), inner_r, color, 1, cv2.LINE_AA)