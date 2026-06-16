import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    RunningMode,
)

MODELS_DIR = Path(__file__).parent.parent / "models"

FINGER_TIP_IDS = [4, 8, 12, 16, 20]
FINGER_PIP_IDS = [3, 6, 10, 14, 18]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


@dataclass
class HandData:
    landmarks: np.ndarray
    world_landmarks: np.ndarray
    handedness: str
    confidence: float
    bounding_box: tuple
    finger_states: dict
    gesture_label: str = "unknown"
    gesture_score: float = 0.0


class HandTracker:
    def __init__(self, config: dict):
        mp_config = config.get("mediapipe", {})
        self._timestamp_ms = 0

        gesture_model = str(MODELS_DIR / "gesture_recognizer.task")

        gesture_options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=gesture_model),
            running_mode=RunningMode.VIDEO,
            num_hands=mp_config.get("max_num_hands", 2),
            min_hand_detection_confidence=mp_config.get("min_detection_confidence", 0.7),
            min_hand_presence_confidence=mp_config.get("min_tracking_confidence", 0.5),
            min_tracking_confidence=mp_config.get("min_tracking_confidence", 0.5),
        )
        self.recognizer = GestureRecognizer.create_from_options(gesture_options)

    def process(self, frame: np.ndarray) -> list[HandData]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = _to_mp_image(rgb_frame)
        self._timestamp_ms += 33

        result = self.recognizer.recognize_for_video(mp_image, self._timestamp_ms)

        hands_data = []
        if result.hand_landmarks and result.handedness:
            for i, (landmarks_proto, handedness_proto) in enumerate(
                zip(result.hand_landmarks, result.handedness)
            ):
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_proto])

                world_landmarks = np.zeros_like(landmarks)
                if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                    world_landmarks = np.array(
                        [[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[i]]
                    )

                handedness = handedness_proto[0].category_name
                confidence = handedness_proto[0].score

                bounding_box = self._compute_bounding_box(landmarks, frame.shape)
                finger_states = self._detect_finger_states(landmarks, handedness)

                gesture_label = "unknown"
                gesture_score = 0.0
                if result.gestures and i < len(result.gestures) and result.gestures[i]:
                    top = result.gestures[i][0]
                    gesture_label = top.category_name
                    gesture_score = top.score

                hands_data.append(HandData(
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    handedness=handedness,
                    confidence=confidence,
                    bounding_box=bounding_box,
                    finger_states=finger_states,
                    gesture_label=gesture_label,
                    gesture_score=gesture_score,
                ))

        return hands_data

    def _compute_bounding_box(self, landmarks: np.ndarray, frame_shape: tuple) -> tuple:
        h, w = frame_shape[:2]
        x_coords = landmarks[:, 0] * w
        y_coords = landmarks[:, 1] * h
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        return (x_min, y_min, x_max, y_max)

    def _detect_finger_states(self, landmarks: np.ndarray, handedness: str) -> dict:
        states = {}
        is_right = handedness == "Right"

        thumb_tip = landmarks[FINGER_TIP_IDS[0]]
        thumb_ip = landmarks[FINGER_PIP_IDS[0]]
        if is_right:
            states["thumb"] = "extended" if thumb_tip[0] < thumb_ip[0] else "curled"
        else:
            states["thumb"] = "extended" if thumb_tip[0] > thumb_ip[0] else "curled"

        for i in range(1, 5):
            tip = landmarks[FINGER_TIP_IDS[i]]
            pip = landmarks[FINGER_PIP_IDS[i]]
            states[FINGER_NAMES[i]] = "extended" if tip[1] < pip[1] else "curled"

        return states

    def release(self):
        self.recognizer.close()


def _to_mp_image(rgb_array: np.ndarray):
    from mediapipe import Image, ImageFormat
    return Image(image_format=ImageFormat.SRGB, data=rgb_array)
