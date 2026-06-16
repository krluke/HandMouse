import numpy as np
from src.hand_tracker import HandData


PINKY_TIP = 20


class CursorMapper:
    def __init__(self, config: dict):
        pipe_config = config.get("pipeline", {})
        self.smoothing: float = pipe_config.get("cursor_smoothing", 0.3)
        self._prev_x: float = 0.5
        self._prev_y: float = 0.5

    def map_to_cursor(self, hands: list[HandData], confidence_threshold: float = 0.7) -> tuple[float, float, bool]:
        if not hands:
            return self._prev_x, self._prev_y, False

        hand = hands[0]
        if hand.confidence < confidence_threshold:
            return self._prev_x, self._prev_y, False

        lm = hand.landmarks
        pinky_tip = lm[PINKY_TIP]
        x = float(np.clip(pinky_tip[0], 0.0, 1.0))
        y = float(np.clip(pinky_tip[1], 0.0, 1.0))

        sx = self._prev_x + self.smoothing * (x - self._prev_x)
        sy = self._prev_y + self.smoothing * (y - self._prev_y)

        self._prev_x = sx
        self._prev_y = sy

        return sx, sy, True