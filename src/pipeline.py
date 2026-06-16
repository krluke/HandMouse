import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional
from src.hand_tracker import HandTracker, HandData
from src.gesture_llm import GestureLLM
from src.mouse.detector import MouseSimulator
from src.mouse.state import MouseState

logger = logging.getLogger(__name__)


BASIC_GESTURES = {
    (False, False, False, False, False): "fist",
    (True, True, True, True, True): "open_palm",
    (False, True, True, False, False): "peace_sign",
    (True, False, False, False, False): "thumbs_up",
    (False, True, False, False, False): "pointing_up",
    (True, False, False, False, True): "call_me",
    (False, True, True, True, True): "three",
    (False, True, True, True, False): "three",
    (True, True, False, False, True): "rock_sign",
}


@dataclass
class PipelineResult:
    hands: list[HandData]
    gesture_label: str
    gesture_source: str
    gesture_confidence: float
    gesture_description: str


class Pipeline:
    def __init__(self, config: dict):
        self.config = config
        self.hand_tracker = HandTracker(config)
        self.gesture_llm = GestureLLM(config)
        self.mouse_sim = MouseSimulator(config)
        self._nim_enabled = bool(config.get("nim", {}).get("api_key"))
        pipe_config = config.get("pipeline", {})
        self.change_threshold = pipe_config.get("landmark_change_threshold", 0.03)
        self.debounce_frames = pipe_config.get("debounce_frames", 5)
        self._prev_landmarks: Optional[np.ndarray] = None
        self._debounce_counter = 0
        self._current_gesture = {
            "gesture": "unknown",
            "confidence": 0.0,
            "description": "",
        }
        self._gesture_source = "none"

    def process(self, frame: np.ndarray) -> tuple[PipelineResult, MouseState]:
        hands = self.hand_tracker.process(frame)
        mouse_state = self.mouse_sim.update(hands)

        mp_gesture = hands[0].gesture_label if hands and hands[0].gesture_label != "None" else None
        mp_score = hands[0].gesture_score if hands else 0.0
        fallback_gesture = self._classify_local(hands) if hands else "no_hand"

        if not hands:
            self._current_gesture = {
                "gesture": "no_hand",
                "confidence": 1.0,
                "description": "No hand detected",
            }
            self._gesture_source = "none"
            self._prev_landmarks = None
        else:
            pose_changed = self._has_pose_changed(hands)

            if mp_gesture and mp_gesture != "None":
                self._current_gesture = {
                    "gesture": mp_gesture,
                    "confidence": mp_score,
                    "description": f"MediaPipe classified as {mp_gesture}",
                }
                self._gesture_source = "mediapipe"

            elif fallback_gesture != "unknown":
                self._current_gesture = {
                    "gesture": fallback_gesture,
                    "confidence": 0.8,
                    "description": f"Locally classified as {fallback_gesture}",
                }
                self._gesture_source = "local"

            if self._nim_enabled and pose_changed and self._debounce_counter <= 0:
                llm_gesture = self.gesture_llm.classify_gesture(frame)
                if llm_gesture and llm_gesture.get("confidence", 0) > 0.5:
                    self._current_gesture = llm_gesture
                    self._gesture_source = "nim"
                    self._debounce_counter = self.debounce_frames

            if self._debounce_counter > 0:
                self._debounce_counter -= 1

        return PipelineResult(
            hands=hands,
            gesture_label=self._current_gesture.get("gesture", "unknown"),
            gesture_source=self._gesture_source,
            gesture_confidence=self._current_gesture.get("confidence", 0.0),
            gesture_description=self._current_gesture.get("description", ""),
        ), mouse_state

    def _classify_local(self, hands: list[HandData]) -> str:
        if not hands:
            return "no_hand"

        hand = hands[0]
        states = hand.finger_states
        key = tuple(
            states.get(finger, "curled") == "extended"
            for finger in ["thumb", "index", "middle", "ring", "pinky"]
        )
        return BASIC_GESTURES.get(key, "unknown")

    def _has_pose_changed(self, hands: list[HandData]) -> bool:
        if not hands:
            self._prev_landmarks = None
            return False

        current = hands[0].landmarks
        if self._prev_landmarks is None:
            self._prev_landmarks = current.copy()
            return True

        diff = np.abs(current - self._prev_landmarks).mean()
        changed = diff > self.change_threshold
        if changed:
            self._prev_landmarks = current.copy()
        return changed

    def release(self):
        self.hand_tracker.release()
