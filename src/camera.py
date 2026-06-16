import cv2
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, config: dict):
        cam_config = config.get("camera", {})
        self.device_index = cam_config.get("device_index", 0)
        self.width = cam_config.get("width", 1280)
        self.height = cam_config.get("height", 720)
        self.fps = cam_config.get("fps", 30)
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            logger.error(f"Failed to open camera at index {self.device_index}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")

        return True

    def read(self) -> Optional[cv2.typing.MatLike]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            return None
        return frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
