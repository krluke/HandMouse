import cv2
import numpy as np
import time
from src.mouse.state import MouseState
from src.os.icons import get_icon_grid, APP_LABELS
from src.os.cursor import draw_cursor
from src.os.clickable import hit_test


TITLE_H = 36
TASKBAR_H = 36


class OSRenderer:
    def __init__(self, config: dict):
        self.config = config
        self._last_icon_hits: dict[str, float] = {}
        self._active_app: str = ""
        self._app_log: list[tuple[str, str]] = []
        self._clock_str = ""
        self._prev_cursor_pos: tuple = (0.5, 0.5)

    def render(
        self,
        w: int,
        h: int,
        mouse_state: MouseState,
        fps: float,
        click_fired: bool,
        right_click_fired: bool,
    ) -> tuple[np.ndarray, list[tuple[str, str]]]:
        canvas = self._draw_background(w, h)

        icons = get_icon_grid(w, h)

        cx = int(mouse_state.cursor_x * w)
        cy = int(mouse_state.cursor_y * h)
        hovered = hit_test(mouse_state.cursor_x, mouse_state.cursor_y, icons, w, h)

        if hovered:
            self._highlight_icon(canvas, hovered)

        if mouse_state.cursor_visible:
            draw_cursor(canvas, cx, cy, True)
        else:
            draw_cursor(canvas, cx, cy, False)

        if click_fired and hovered:
            app_name = APP_LABELS.get(hovered.id, hovered.id.title())
            self._active_app = app_name
            ts = time.strftime("%H:%M:%S")
            entry = (app_name, ts)
            self._app_log.insert(0, entry)
            if len(self._app_log) > 8:
                self._app_log.pop()
            self._last_icon_hits[hovered.id] = time.time()

        if right_click_fired and hovered:
            app_name = APP_LABELS.get(hovered.id, hovered.id.title())
            ts = time.strftime("%H:%M:%S")
            entry = (f"{app_name} [right-click]", ts)
            self._app_log.insert(0, entry)
            if len(self._app_log) > 8:
                self._app_log.pop()

        self._draw_title_bar(canvas, w)
        self._draw_taskbar(canvas, w, h)
        self._draw_app_info(canvas, w, h)

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return canvas_rgb, list(self._app_log)

    def _draw_background(self, w: int, h: int) -> np.ndarray:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        bg = (26, 26, 46)
        canvas[:] = bg
        return canvas

    def _draw_title_bar(self, canvas: np.ndarray, w: int):
        bar = np.zeros((TITLE_H, w, 3), dtype=np.uint8)
        bar[:] = (22, 33, 64)
        canvas[:TITLE_H] = bar

        cv2.putText(canvas, "HandMouse OS", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)

        bw, bh = 18, 18
        gap = 8
        xs = [(w - gap * 3 - 3 * bw), (w - gap * 2 - 2 * bw), (w - gap - bw)]
        ys = [(TITLE_H - bh) // 2] * 3
        colors = [(120, 120, 130), (120, 120, 130), (220, 80, 80)]
        for bx, by, col in zip(xs, ys, colors):
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), col, -1, cv2.LINE_AA)

    def _draw_taskbar(self, canvas: np.ndarray, w: int, h: int):
        y0 = h - TASKBAR_H
        bar = canvas[y0:h]
        bar[:] = (15, 25, 55)

        cv2.rectangle(canvas, (0, y0), (w, h), (30, 45, 80), 1, cv2.LINE_AA)

        cv2.circle(canvas, (20, y0 + TASKBAR_H // 2), 8, (80, 140, 255), -1, cv2.LINE_AA)

        if self._active_app:
            label = self._active_app
            cv2.putText(canvas, label, (36, y0 + TASKBAR_H // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 255), 1, cv2.LINE_AA)

        self._clock_str = time.strftime("%H:%M")
        cv2.putText(canvas, self._clock_str, (w - 50, y0 + TASKBAR_H // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 195, 220), 1, cv2.LINE_AA)

    def _highlight_icon(self, canvas: np.ndarray, icon, alpha=0.5):
        x, y = icon.x, icon.y
        bw, bh = icon.width, icon.height
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x - 3, y - 3), (x + bw + 3, y + bh + 3), (100, 180, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    def _draw_app_info(self, canvas: np.ndarray, w: int, h: int):
        pass

    def get_app_log(self) -> list[tuple[str, str]]:
        return list(self._app_log)