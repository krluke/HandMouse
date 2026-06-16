import requests
import base64
import time
import logging
import cv2
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

GESTURE_PROMPT = (
    "You are a hand gesture recognition assistant. "
    "Given this image of a hand with tracked landmarks, classify the gesture. "
    "Respond with ONLY a JSON object in this exact format, no other text:\n"
    '{"gesture": "<gesture_name>", "confidence": <0.0-1.0>, "description": "<brief description>"}\n\n'
    "Common gestures: fist, open_palm, peace_sign, thumbs_up, thumbs_down, "
    "pointing_up, pointing_forward, ok_sign, rock_sign, pinching, gripping, "
    "swipe_left, swipe_right, waving, call_me, unknown\n\n"
    "If the image shows no clear hand gesture, respond with: "
    '{"gesture": "unknown", "confidence": 0.0, "description": "No clear gesture detected"}'
)


class GestureLLM:
    def __init__(self, config: dict):
        nim_config = config.get("nim", {})
        self.api_url = nim_config.get("api_url", "https://integrate.api.nvidia.com/v1/chat/completions")
        self.model_id = nim_config.get("model_id", "meta/llama-3.2-11b-vision-instruct")
        self.api_key = nim_config.get("api_key", "")
        self.max_tokens = nim_config.get("max_tokens", 256)
        self.temperature = nim_config.get("temperature", 0.2)
        self.top_p = nim_config.get("top_p", 0.7)
        self.min_interval = nim_config.get("min_request_interval_sec", 1.0)
        self.resize_width = nim_config.get("frame_resize_width", 640)
        self.resize_height = nim_config.get("frame_resize_height", 480)
        self._last_request_time = 0.0
        self._last_gesture = {"gesture": "unknown", "confidence": 0.0, "description": ""}

    def classify_gesture(self, frame: np.ndarray) -> Optional[dict]:
        if not self.api_key:
            logger.warning("NIM API key not set. Skipping LLM gesture classification.")
            return None

        now = time.time()
        if now - self._last_request_time < self.min_interval:
            return self._last_gesture

        try:
            b64_image = self._encode_frame(frame)
            response = self._call_nim(b64_image)
            gesture = self._parse_response(response)
            if gesture:
                self._last_gesture = gesture
            self._last_request_time = now
            return gesture or self._last_gesture
        except Exception as e:
            logger.error(f"NIM request failed: {e}")
            return self._last_gesture

    def _encode_frame(self, frame: np.ndarray) -> str:
        resized = cv2.resize(frame, (self.resize_width, self.resize_height))
        _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    def _call_nim(self, b64_image: str) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": f'<img src="data:image/jpeg;base64,{b64_image}" />\n{GESTURE_PROMPT}',
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }

        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _parse_response(self, response: dict) -> Optional[dict]:
        try:
            content = response["choices"][0]["message"]["content"].strip()
            import json

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            gesture = json.loads(content)
            if "gesture" in gesture and "confidence" in gesture:
                return gesture
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to parse NIM response: {e}")
        return None
