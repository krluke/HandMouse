import cv2
import numpy as np
from src.pipeline import PipelineResult

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FINGER_TIPS = {4, 8, 12, 16, 20}
WRIST = 0


class Renderer:
    def __init__(self, config: dict):
        display = config.get("display", {})
        self.show_landmarks = display.get("show_landmarks", True)
        self.show_connections = display.get("show_connections", True)
        self.show_fps = display.get("show_fps", True)

    def render_skeleton(self, frame_shape: tuple, result: PipelineResult, fps: float) -> np.ndarray:
        h, w = frame_shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        for hand_data in result.hands:
            landmarks = hand_data.landmarks

            if self.show_connections:
                for i, j in HAND_CONNECTIONS:
                    x1 = int(landmarks[i][0] * w)
                    y1 = int(landmarks[i][1] * h)
                    x2 = int(landmarks[j][0] * w)
                    y2 = int(landmarks[j][1] * h)
                    color = (180, 180, 180)
                    if i in FINGER_TIPS or j in FINGER_TIPS:
                        color = (200, 255, 0)
                    cv2.line(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            if self.show_landmarks:
                for idx in range(21):
                    x = int(landmarks[idx][0] * w)
                    y = int(landmarks[idx][1] * h)
                    if idx == WRIST:
                        cv2.circle(canvas, (x, y), 4, (255, 200, 0), -1, cv2.LINE_AA)
                    elif idx in FINGER_TIPS:
                        cv2.circle(canvas, (x, y), 4, (100, 255, 0), -1, cv2.LINE_AA)
                    else:
                        cv2.circle(canvas, (x, y), 2, (200, 200, 200), -1, cv2.LINE_AA)

        if self.show_fps:
            cv2.putText(canvas, f"FPS: {fps:.1f}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return canvas_rgb

    def render_with_camera(self, frame: np.ndarray, result: PipelineResult, fps: float) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        for hand_data in result.hands:
            landmarks = hand_data.landmarks

            if self.show_connections:
                for i, j in HAND_CONNECTIONS:
                    x1 = int(landmarks[i][0] * w)
                    y1 = int(landmarks[i][1] * h)
                    x2 = int(landmarks[j][0] * w)
                    y2 = int(landmarks[j][1] * h)
                    color = (180, 180, 180)
                    if i in FINGER_TIPS or j in FINGER_TIPS:
                        color = (200, 255, 0)
                    cv2.line(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            if self.show_landmarks:
                for idx in range(21):
                    x = int(landmarks[idx][0] * w)
                    y = int(landmarks[idx][1] * h)
                    if idx == WRIST:
                        cv2.circle(annotated, (x, y), 4, (255, 200, 0), -1, cv2.LINE_AA)
                    elif idx in FINGER_TIPS:
                        cv2.circle(annotated, (x, y), 4, (100, 255, 0), -1, cv2.LINE_AA)
                    else:
                        cv2.circle(annotated, (x, y), 2, (200, 200, 200), -1, cv2.LINE_AA)

        if self.show_fps:
            cv2.putText(annotated, f"FPS: {fps:.1f}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return annotated_rgb
