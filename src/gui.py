import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QCheckBox, QComboBox, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from src.pipeline import PipelineResult
from src.mouse.state import MouseState

DARK_BG = "#1e1e1e"
ACCENT_GREEN = "#00ff88"
CYAN = "#00ffff"
DIM = "#666666"
YELLOW = "#ffcc00"
MAGENTA = "#ff66ff"
LIGHT_GRAY = "#cccccc"

SOURCE_COLORS = {"mediapipe": ACCENT_GREEN, "nim": MAGENTA, "local": YELLOW, "none": DIM}
SOURCE_DISPLAY = {"mediapipe": "MEDIAPIPE", "nim": "NIM (LLM)", "local": "LOCAL RULES", "none": "—"}


class HandTrackerGUI:
    def __init__(self, config: dict, on_close_callback=None):
        self.config = config
        self.on_close = on_close_callback

        self._app = None
        self._window = None
        self._video_label = None
        self._fps_label = None
        self._gesture_label = None
        self._confidence_label = None
        self._source_label = None
        self._hand_info_label = None
        self._finger_label = None
        self._position_label = None
        self._nim_checkbox = None
        self._display_mode_combo = None
        self._app_log_label = None
        self._app_log: list = []
        self._status_label = None
        self._running = False

    def build(self):
        self._app = QApplication.instance() or QApplication([])
        self._app.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {DARK_BG}; }}
            QLabel {{ color: #ffffff; font-family: Consolas, monospace; font-size: 11pt; }}
            QCheckBox {{ color: #ffffff; font-family: Consolas, monospace; font-size: 10pt; }}
            QCheckBox::indicator {{ width: 14px; height: 14px; }}
            QComboBox {{ color: #ffffff; background-color: #2a2a2a; font-family: Consolas, monospace; font-size: 10pt; border: 1px solid #444444; padding: 2px 6px; }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox::down-arrow {{ image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #888888; margin-right: 4px; }}
            QComboBox QAbstractItemView {{ background-color: #2a2a2a; color: #ffffff; selection-background-color: {ACCENT_GREEN}; }}
            QFrame#separator {{ background-color: #333333; max-height: 1px; }}
        """)

        self._window = QMainWindow()
        self._window.setWindowTitle("HandTracker — Real-time Hand Tracking & Gesture Recognition")
        self._window.setMinimumSize(1100, 650)

        central = QWidget()
        self._window.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        video_frame = QFrame()
        video_frame.setStyleSheet("background-color: #000000; border: 1px solid #333333;")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_label.setMinimumSize(640, 360)
        video_layout.addWidget(self._video_label)
        main_layout.addWidget(video_frame, stretch=3)

        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        sidebar_layout.setSpacing(4)

        title = QLabel("HAND TRACKER")
        title.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 13pt; font-weight: bold;")
        sidebar_layout.addWidget(title)
        sidebar_layout.addSpacing(12)

        self._fps_label = QLabel("FPS: --")
        sidebar_layout.addWidget(self._fps_label)
        sidebar_layout.addSpacing(8)

        sidebar_layout.addWidget(self._make_separator())
        sidebar_layout.addWidget(QLabel("GESTURE"))

        self._gesture_label = QLabel("—")
        self._gesture_label.setStyleSheet(f"color: {CYAN}; font-size: 18pt; font-weight: bold;")
        sidebar_layout.addWidget(self._gesture_label)

        self._confidence_label = QLabel("Confidence: —")
        sidebar_layout.addWidget(self._confidence_label)

        self._source_label = QLabel("Source: —")
        self._source_label.setStyleSheet(f"color: {YELLOW}; font-weight: bold; font-size: 10pt;")
        sidebar_layout.addWidget(self._source_label)
        sidebar_layout.addSpacing(10)

        sidebar_layout.addWidget(self._make_separator())
        sidebar_layout.addWidget(QLabel("HAND INFO"))
        self._hand_info_label = QLabel("No hand detected")
        sidebar_layout.addWidget(self._hand_info_label)

        self._position_label = QLabel("Position: —")
        sidebar_layout.addWidget(self._position_label)
        sidebar_layout.addSpacing(6)

        sidebar_layout.addWidget(self._make_separator())
        sidebar_layout.addWidget(QLabel("FINGERS"))
        self._finger_label = QLabel("—")
        self._finger_label.setStyleSheet(f"color: {LIGHT_GRAY}; font-size: 10pt;")
        sidebar_layout.addWidget(self._finger_label)
        sidebar_layout.addSpacing(6)

        sidebar_layout.addWidget(self._make_separator())

        nim_enabled = bool(self.config.get("nim", {}).get("api_key"))
        self._nim_checkbox = QCheckBox("Enable NIM (Llama 3.2 Vision)")
        self._nim_checkbox.setChecked(nim_enabled)
        sidebar_layout.addWidget(self._nim_checkbox)
        sidebar_layout.addSpacing(6)

        sidebar_layout.addWidget(QLabel("DISPLAY MODE"))
        self._display_mode_combo = QComboBox()
        self._display_mode_combo.addItems(["Skeleton Only", "Camera + Skeleton", "OS Screen"])
        self._display_mode_combo.setCurrentIndex(0)
        sidebar_layout.addWidget(self._display_mode_combo)
        sidebar_layout.addSpacing(6)

        sidebar_layout.addWidget(self._make_separator())
        sidebar_layout.addWidget(QLabel("APP LOG"))
        self._app_log_label = QLabel("No apps opened yet")
        self._app_log_label.setStyleSheet(f"color: {DIM}; font-size: 9pt;")
        self._app_log_label.setWordWrap(True)
        sidebar_layout.addWidget(self._app_log_label)
        sidebar_layout.addSpacing(10)

        self._status_label = QLabel("Status: Running")
        self._status_label.setStyleSheet(f"color: {YELLOW}; font-weight: bold; font-size: 10pt;")
        sidebar_layout.addWidget(self._status_label)

        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar, stretch=0)

        self._window.closeEvent = lambda event: self._on_close()

    def show_frame(self, frame: np.ndarray):
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).copy()

        label_w = self._video_label.width()
        label_h = self._video_label.height()
        if label_w < 2 or label_h < 2:
            label_w = 640
            label_h = 360

        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.FastTransformation)
        self._video_label.setPixmap(scaled)

    def update_info(self, result: PipelineResult, fps: float, mouse_state: MouseState = None,
                    log_messages: list = None):
        self._fps_label.setText(f"FPS: {fps:.1f}")

        if mouse_state is not None:
            if mouse_state.cursor_visible:
                cx = int(mouse_state.cursor_x * 100)
                cy = int(mouse_state.cursor_y * 100)
                self._position_label.setText(f"Cursor: ({cx}%, {cy}%)")
                status = []
                if mouse_state.pick:
                    status.append("DROP")
                if mouse_state.is_dragging:
                    status.append("DRAG")
                if mouse_state.left_click:
                    status.append("L-CLICK")
                if mouse_state.right_click:
                    status.append("R-CLICK")
                if mouse_state.scroll_dy != 0:
                    status.append(f"SCROLL {mouse_state.scroll_dy:+.2f}")
                self._status_label.setText(" | ".join(status) if status else "Cursor active")
            else:
                self._position_label.setText("Cursor: hidden")
                self._status_label.setText("Cursor hidden")

        if log_messages is not None and len(log_messages) > 0:
            self._app_log = list(log_messages)
            log_text = "\n".join([f"{name}  @ {ts}" for name, ts in self._app_log[:5]])
            self._app_log_label.setText(log_text if log_text else "No apps opened yet")
            self._app_log_label.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 9pt;")

        gesture = result.gesture_label
        if gesture == "no_hand":
            self._gesture_label.setText("No Hand")
            self._gesture_label.setStyleSheet(f"color: {DIM}; font-size: 18pt; font-weight: bold;")
            self._confidence_label.setText("Confidence: —")
            self._source_label.setText("Source: —")
            self._hand_info_label.setText("No hand detected")
            if mouse_state is None or not mouse_state.cursor_visible:
                self._position_label.setText("Position: —")
            self._finger_label.setText("—")
            return

        display_name = gesture.replace("_", " ").title()
        self._gesture_label.setText(display_name)
        self._gesture_label.setStyleSheet(f"color: {CYAN}; font-size: 18pt; font-weight: bold;")
        self._confidence_label.setText(f"Confidence: {result.gesture_confidence:.0%}")

        color = SOURCE_COLORS.get(result.gesture_source, "#ffffff")
        tag = SOURCE_DISPLAY.get(result.gesture_source, result.gesture_source.upper())
        self._source_label.setText(f"Source: {tag}")
        self._source_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 10pt;")

        if result.hands:
            hand = result.hands[0]
            self._hand_info_label.setText(f"{hand.handedness} hand ({hand.confidence:.0%})")

            finger_parts = []
            for name, state in hand.finger_states.items():
                symbol = "▲" if state == "extended" else "▼"
                finger_parts.append(f"{name[0].upper()}:{symbol}")
            self._finger_label.setText(" ".join(finger_parts))
        else:
            self._hand_info_label.setText("No hand detected")
            self._finger_label.setText("—")

    def is_nim_enabled(self) -> bool:
        if self._nim_checkbox is None:
            return False
        return self._nim_checkbox.isChecked()

    def get_display_mode(self) -> int:
        if self._display_mode_combo is None:
            return 0
        return self._display_mode_combo.currentIndex()

    def is_running(self) -> bool:
        return self._running

    def start(self):
        self._running = True
        if self._window:
            self._window.show()
        if self._app:
            self._app.exec()

    def stop(self):
        self._running = False
        if self._app:
            self._app.quit()

    def schedule_callback(self, func, delay_ms: int):
        if self._app:
            QTimer.singleShot(delay_ms, func)

    def _on_close(self):
        self._running = False
        if self.on_close:
            self.on_close()
        if self._app:
            self._app.quit()

    @staticmethod
    def _make_separator() -> QFrame:
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        sep.setFixedHeight(1)
        return sep