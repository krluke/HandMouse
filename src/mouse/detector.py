import time
import numpy as np
from src.hand_tracker import HandData
from src.mouse.state import MouseState, GestureAction
from src.mouse.cursor import CursorMapper

THUMB_IDX = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
WRIST = 0


class MouseSimulator:
    def __init__(self, config: dict):
        self.config = config
        pipe = config.get("pipeline", {})
        self.cursor_mapper = CursorMapper(config)

        self.click_window_ms: float = pipe.get("click_window_ms", 250)
        self.right_click_window_ms: float = pipe.get("right_click_window_ms", 200)
        self.pick_distance_threshold: float = pipe.get("pick_distance_threshold", 0.05)
        self.scroll_threshold: float = pipe.get("scroll_threshold", 0.01)
        self.missed_frames_to_hide: int = pipe.get("missed_frames_to_hide", 3)
        self.drag_move_threshold: float = pipe.get("drag_move_threshold", 0.08)

        self._state = MouseState()
        self._prev_index_extended: bool = True
        self._prev_middle_extended: bool = True
        self._prev_thumb_extended: bool = True
        self._index_curl_time: float = 0.0
        self._right_click_start: float = 0.0
        self._right_click_fired: bool = False
        self._pick_origin: tuple = (0.0, 0.0)
        self._picking: bool = False
        self._prev_scroll_y: float = 0.0
        self._scroll_accum: float = 0.0
        self._missed_frames: int = 0
        self._last_action_time: float = 0.0

    def update(self, hands: list[HandData]) -> MouseState:
        self._state.clear_events()

        cx, cy, visible = self.cursor_mapper.map_to_cursor(hands, confidence_threshold=0.7)
        if not visible:
            self._missed_frames += 1
            if self._missed_frames >= self.missed_frames_to_hide:
                self._state.cursor_visible = False
        else:
            self._missed_frames = 0
            self._state.cursor_x = cx
            self._state.cursor_y = cy
            self._state.cursor_visible = True

        self._state.confidence = hands[0].confidence if hands else 0.0

        if not hands:
            self._reset_gesture_tracking()
            return self._state

        hand = hands[0]
        lm = hand.landmarks
        states = hand.finger_states

        index_ext = states.get("index", "curled") == "extended"
        middle_ext = states.get("middle", "curled") == "extended"
        thumb_ext = states.get("thumb", "curled") == "extended"

        self._detect_click(index_ext, hand)
        self._detect_right_click(index_ext, middle_ext, hand)
        self._detect_pick_drop(index_ext, middle_ext, thumb_ext, lm, hand)
        self._detect_scroll(middle_ext, lm, hand)

        self._prev_index_extended = index_ext
        self._prev_middle_extended = middle_ext
        self._prev_thumb_extended = thumb_ext

        return self._state

    def _detect_click(self, index_ext: bool, hand: HandData):
        now = time.time()
        if not self._prev_index_extended and index_ext:
            if now - self._index_curl_time < self.click_window_ms / 1000.0:
                self._state.left_click = True
                self._last_action_time = now
        if not index_ext and self._prev_index_extended:
            self._index_curl_time = now

    def _detect_right_click(self, index_ext: bool, middle_ext: bool, hand: HandData):
        now = time.time()
        if not index_ext and not middle_ext:
            if not self._right_click_fired:
                if self._right_click_start == 0.0:
                    self._right_click_start = now
                elif now - self._right_click_start >= self.right_click_window_ms / 1000.0:
                    self._state.right_click = True
                    self._right_click_fired = True
                    self._last_action_time = now
        else:
            self._right_click_start = 0.0
            self._right_click_fired = False

    def _detect_pick_drop(self, index_ext: bool, middle_ext: bool, thumb_ext: bool, lm, hand: HandData):
        is_picking_now = index_ext and middle_ext and thumb_ext and self._thumb_close_to_index(lm)

        if is_picking_now and not self._picking:
            self._picking = True
            self._pick_origin = (float(lm[INDEX_TIP][0]), float(lm[INDEX_TIP][1]))
            self._state.is_picking = True
            self._state.is_dragging = False
            self._last_action_time = time.time()

        elif is_picking_now and self._picking:
            if self._was_dragging(lm):
                self._state.is_dragging = True

        elif not is_picking_now and self._picking:
            was_dragging = self._state.is_dragging
            self._picking = False
            self._state.is_picking = False
            self._state.is_dragging = False
            self._last_action_time = time.time()

    def _thumb_close_to_index(self, lm) -> bool:
        thumb = lm[THUMB_IDX]
        idx = lm[INDEX_TIP]
        dist = np.sqrt((thumb[0] - idx[0]) ** 2 + (thumb[1] - idx[1]) ** 2)
        return dist < self.pick_distance_threshold

    def _was_dragging(self, lm) -> bool:
        cur = (float(lm[INDEX_TIP][0]), float(lm[INDEX_TIP][1]))
        dist = np.sqrt((cur[0] - self._pick_origin[0]) ** 2 + (cur[1] - self._pick_origin[1]) ** 2)
        return dist > self.drag_move_threshold

    def _detect_scroll(self, middle_ext: bool, lm, hand: HandData):
        if not middle_ext:
            self._prev_scroll_y = 0.0
            self._scroll_accum = 0.0
            return

        cur_y = float(lm[MIDDLE_TIP][1])
        if self._prev_scroll_y == 0.0:
            self._prev_scroll_y = cur_y
            return

        delta = self._prev_scroll_y - cur_y
        if abs(delta) > self.scroll_threshold:
            self._scroll_accum += delta
            self._state.scroll_dy = self._scroll_accum
            self._last_action_time = time.time()

        self._prev_scroll_y = cur_y

    def _reset_gesture_tracking(self):
        self._picking = False
        self._right_click_start = 0.0
        self._right_click_fired = False
        self._index_curl_time = 0.0
        self._prev_scroll_y = 0.0
        self._scroll_accum = 0.0