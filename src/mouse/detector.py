import time
import numpy as np
from collections import deque
from src.hand_tracker import HandData
from src.mouse.state import MouseState
from src.mouse.cursor import CursorMapper, PINKY_TIP

PINKY_PIP = 18
PINKY_DIP = 19


class MouseSimulator:
    def __init__(self, config: dict):
        self.config = config
        pipe = config.get("pipeline", {})
        self.cursor_mapper = CursorMapper(config)

        self.click_window_ms: float = pipe.get("click_window_ms", 300)
        self.right_click_window_ms: float = pipe.get("right_click_window_ms", 300)
        self.pinky_dip_threshold_deg: float = pipe.get("pinky_dip_threshold_deg", 15.0)
        self.pinky_bend_min_frames: int = pipe.get("pinky_bend_min_frames", 3)
        self.missed_frames_to_hide: int = pipe.get("missed_frames_to_hide", 3)
        self.scroll_drag_threshold: float = pipe.get("scroll_drag_threshold", 0.06)
        self.pinky_scroll_scale: float = pipe.get("pinky_scroll_scale", 2.0)

        self._state = MouseState()

        self._prev_pinky_dip_angle: float = 0.0
        self._pinky_dip_angles: deque[float] = deque(maxlen=15)
        self._pinky_dip_delta_history: deque[float] = deque(maxlen=15)
        self._bend_count: int = 0
        self._was_scrolling_or_dragging: bool = False

        self._prev_index_extended: bool = False
        self._prev_middle_extended: bool = False
        self._index_curl_start: float = 0.0
        self._middle_curl_start: float = 0.0
        self._index_was_fully_curled: bool = False
        self._middle_was_fully_curled: bool = False
        self._index_ever_extended: bool = False
        self._middle_ever_extended: bool = False

        self._missed_frames: int = 0
        self._scroll_start_pos: tuple[float, float] = (0.0, 0.0)
        self._scroll_origin_valid: bool = False

    def update(self, hands: list[HandData]) -> MouseState:
        self._state.clear_events()
        self._state.is_scrolling = False
        self._state.is_dragging = False
        self._state.pick = False

        if not hands:
            self._missed_frames += 1
            if self._missed_frames >= self.missed_frames_to_hide:
                self._state.cursor_visible = False
            self._reset_scroll_state()
            return self._state

        self._missed_frames = 0
        hand = hands[0]
        lm = hand.landmarks
        states = hand.finger_states
        confidence = hand.confidence

        index_ext = states.get("index", "curled") == "extended"
        middle_ext = states.get("middle", "curled") == "extended"
        thumb_ext = states.get("thumb", "curled") == "extended"
        pinky_ext = states.get("pinky", "curled") == "extended"

        pinky_dip_angle = self._compute_pinky_dip_angle(lm)

        if pinky_ext and confidence >= 0.6:
            self._state.cursor_visible = True
            cx, cy, _ = self.cursor_mapper.map_to_cursor(hands, confidence_threshold=0.6)
            self._state.cursor_x = cx
            self._state.cursor_y = cy

            if self._was_scrolling_or_dragging:
                self._state.pick = True
                self._was_scrolling_or_dragging = False
                self._reset_scroll_state()

        else:
            self._state.cursor_visible = False

        self._detect_left_click(index_ext)
        self._detect_right_click(middle_ext)
        self._detect_scroll_or_drag(pinky_ext, pinky_dip_angle, lm)

        self._prev_index_extended = index_ext
        self._prev_middle_extended = middle_ext
        self._prev_pinky_dip_angle = pinky_dip_angle

        return self._state

    def _compute_pinky_dip_angle(self, lm) -> float:
        pip = lm[PINKY_PIP]
        dip = lm[PINKY_DIP]
        angle = np.arctan2(dip[1] - pip[1], dip[0] - pip[0])
        return angle

    def _detect_left_click(self, index_ext: bool):
        now = time.time()

        if index_ext:
            self._index_ever_extended = True

        if self._index_ever_extended and self._prev_index_extended and not index_ext:
            self._index_curl_start = now
            self._index_was_fully_curled = True

        if self._index_was_fully_curled and index_ext:
            elapsed = now - self._index_curl_start
            if elapsed < self.click_window_ms / 1000.0:
                self._state.left_click = True
            self._index_was_fully_curled = False
            self._index_ever_extended = False

    def _detect_right_click(self, middle_ext: bool):
        now = time.time()

        if middle_ext:
            self._middle_ever_extended = True

        if self._middle_ever_extended and self._prev_middle_extended and not middle_ext:
            self._middle_curl_start = now
            self._middle_was_fully_curled = True

        if self._middle_was_fully_curled and middle_ext:
            elapsed = now - self._middle_curl_start
            if elapsed < self.right_click_window_ms / 1000.0:
                self._state.right_click = True
            self._middle_was_fully_curled = False
            self._middle_ever_extended = False

    def _detect_scroll_or_drag(self, pinky_ext: bool, pinky_dip_angle: float,
                               lm):
        self._pinky_dip_angles.append(pinky_dip_angle)

        if len(self._pinky_dip_angles) < 2:
            return

        delta = abs(pinky_dip_angle - self._prev_pinky_dip_angle)
        self._pinky_dip_delta_history.append(delta)

        if not pinky_ext:
            self._bend_count = 0
            self._was_scrolling_or_dragging = False
            return

        threshold_rad = np.deg2rad(self.pinky_dip_threshold_deg)
        recent_deltas = list(self._pinky_dip_delta_history)[-6:]
        bent_frames = sum(1 for d in recent_deltas if d > threshold_rad)

        if bent_frames >= 2:
            self._bend_count += 1
        else:
            self._bend_count = max(0, self._bend_count - 1)

        if self._bend_count >= self.pinky_bend_min_frames:
            if not self._was_scrolling_or_dragging:
                self._scroll_origin_valid = True
                self._was_scrolling_or_dragging = True

            scroll_delta = 0.0
            if len(self._pinky_dip_delta_history) >= 3:
                scroll_delta = sum(list(self._pinky_dip_delta_history)[-3:])

            self._state.scroll_dy = scroll_delta * self.pinky_scroll_scale
            self._state.is_scrolling = True

            pinky_tip = lm[PINKY_TIP]
            current_pos = (pinky_tip[0], pinky_tip[1])
            if self._scroll_origin_valid:
                disp = np.sqrt(
                    (current_pos[0] - self._scroll_start_pos[0]) ** 2 +
                    (current_pos[1] - self._scroll_start_pos[1]) ** 2
                )
                if disp > self.scroll_drag_threshold:
                    self._state.is_scrolling = False
                    self._state.is_dragging = True
                    self._scroll_origin_valid = False

        if self._was_scrolling_or_dragging and self._bend_count == 0:
            self._bend_count = 0
            self._pinky_dip_delta_history.clear()
            self._was_scrolling_or_dragging = False
            self._scroll_origin_valid = False

    def _reset_scroll_state(self):
        self._bend_count = 0
        self._was_scrolling_or_dragging = False
        self._scroll_origin_valid = False
        self._pinky_dip_angles.clear()
        self._pinky_dip_delta_history.clear()
        self._index_was_fully_curled = False
        self._middle_was_fully_curled = False
        self._index_curl_start = 0.0
        self._middle_curl_start = 0.0
        self._index_ever_extended = False
        self._middle_ever_extended = False