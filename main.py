import os
import time
import logging
import argparse
import cv2
import numpy as np
from PySide6.QtCore import QTimer
from src.utils import load_config
from src.camera import Camera
from src.pipeline import Pipeline
from src.renderer import Renderer
from src.gui import HandTrackerGUI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FRAME_INTERVAL_MS = 16


def main():
    if not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "wayland"

    parser = argparse.ArgumentParser(description="HandTracker — Real-time hand tracking & gesture recognition")
    parser.add_argument("--config", type=str, default=None, help="Path to settings.yaml")
    parser.add_argument("--nim", action="store_true", help="Enable NIM LLM calls (requires API key in config)")
    args = parser.parse_args()

    config = load_config() if args.config is None else load_config(args.config)

    if args.nim and config.get("nim", {}).get("api_key"):
        logger.info("NIM enabled")
    else:
        if not config.get("nim", {}).get("api_key"):
            logger.info("NIM disabled — no API key found")
        else:
            logger.info("NIM disabled (use --nim to enable)")
        config["nim"]["api_key"] = ""

    camera = Camera(config)
    if not camera.open():
        logger.error("Cannot open camera. Exiting.")
        return

    pipeline = Pipeline(config)
    renderer = Renderer(config)

    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    for _ in range(3):
        pipeline.process(dummy)
    logger.info("MediaPipe model pre-warmed")

    gui = HandTrackerGUI(config, on_close_callback=lambda: None)

    def on_close():
        gui.stop()
        pipeline.release()
        camera.release()
        logger.info("HandTracker stopped.")

    gui.on_close = on_close
    gui.build()

    prev_time = time.time()
    last_result = None
    last_mouse_state = None
    last_annotated = None
    last_fps = 0.0
    last_log: list = []

    def update_loop():
        nonlocal prev_time, last_result, last_annotated, last_fps, last_mouse_state, last_log

        if not gui.is_running():
            return

        frame = camera.read()
        if frame is None:
            QTimer.singleShot(FRAME_INTERVAL_MS, update_loop)
            return
        frame = cv2.flip(frame, 1)

        result, mouse_state = pipeline.process(frame)

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        last_fps = last_fps * 0.9 + (1.0 / dt if dt > 0 else 0) * 0.1

        last_result = result
        last_mouse_state = mouse_state

        display_mode = gui.get_display_mode()
        if display_mode == 1:
            last_annotated = renderer.render_with_camera(frame, result, last_fps)
            last_log = []
        elif display_mode == 2:
            last_annotated, last_log = renderer.render_os_screen(frame.shape, mouse_state, result, last_fps)
        else:
            last_annotated = renderer.render_skeleton(frame.shape, result, last_fps)
            last_log = []

        current_nim = gui.is_nim_enabled()
        if current_nim != bool(config.get("nim", {}).get("api_key")):
            if current_nim:
                config["nim"]["api_key"] = config["nim"].get("_saved_api_key", "")
            else:
                config["nim"]["_saved_api_key"] = config["nim"].get("api_key", "")
                config["nim"]["api_key"] = ""
            pipeline._nim_enabled = current_nim

        gui.show_frame(last_annotated)
        gui.update_info(last_result, last_fps, mouse_state=last_mouse_state, log_messages=last_log)

        elapsed = (time.time() - curr_time) * 1000
        next_interval = max(1, FRAME_INTERVAL_MS - int(elapsed))
        QTimer.singleShot(next_interval, update_loop)

    logger.info("HandTracker running. Close window to quit.")
    QTimer.singleShot(100, update_loop)
    gui.start()


if __name__ == "__main__":
    main()