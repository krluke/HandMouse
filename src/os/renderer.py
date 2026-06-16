import cv2
import numpy as np
import time
from src.mouse.state import MouseState
from src.os.icons import get_icon_grid, IconDef
from src.os.cursor import draw_cursor
from src.os.clickable import hit_test
from src.os.draw_icons import draw_icon_shape


TITLE_H = 36
TASKBAR_H = 36


class OSRenderer:
    def __init__(self, config: dict):
        self.config = config
        self._active_app: str = ""
        self._app_log: list[tuple[str, str]] = []

        self._icon_positions: dict[str, tuple[int, int]] = {}
        self._dragged_icon_id: str | None = None
        self._drag_offset: tuple[int, int] = (0, 0)

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
        for icon in icons:
            if icon.id not in self._icon_positions:
                self._icon_positions[icon.id] = (icon.x, icon.y)

        if mouse_state.is_dragging and self._dragged_icon_id is None:
            cx_n, cy_n = mouse_state.cursor_x, mouse_state.cursor_y
            for icon in icons:
                if icon.id in self._icon_positions:
                    ix, iy = self._icon_positions[icon.id]
                    if ix <= int(cx_n * w) <= ix + icon.width and iy <= int(cy_n * h) <= iy + icon.height:
                        self._dragged_icon_id = icon.id
                        self._drag_offset = (ix - int(cx_n * w), iy - int(cy_n * h))
                        break

        if mouse_state.is_dragging and self._dragged_icon_id is not None:
            nx = int(mouse_state.cursor_x * w) + self._drag_offset[0]
            ny = int(mouse_state.cursor_y * h) + self._drag_offset[1]
            self._icon_positions[self._dragged_icon_id] = (nx, ny)
        elif not mouse_state.is_dragging and self._dragged_icon_id is not None:
            old_pos = self._icon_positions.get(self._dragged_icon_id, (0, 0))
            from src.os.icons import APP_LABELS
            name = APP_LABELS.get(self._dragged_icon_id, self._dragged_icon_id.title())
            ts = time.strftime("%H:%M:%S")
            entry = (f"{name} dropped at ({old_pos[0]}, {old_pos[1]})", ts)
            self._app_log.insert(0, entry)
            if len(self._app_log) > 8:
                self._app_log.pop()
            self._dragged_icon_id = None
            self._drag_offset = (0, 0)

        hovered = hit_test(mouse_state.cursor_x, mouse_state.cursor_y,
                           self._icon_positions, icons, w, h)

        self._draw_icons(canvas, icons, w, h, hovered)

        if mouse_state.cursor_visible:
            cx = int(mouse_state.cursor_x * w)
            cy = int(mouse_state.cursor_y * h)
            draw_cursor(canvas, cx, cy, True)
        else:
            draw_cursor(canvas, 0, 0, False)

        if click_fired and hovered:
            from src.os.icons import APP_LABELS
            self._dragged_icon_id = None
            self._drag_offset = (0, 0)
            app_name = APP_LABELS.get(hovered.id, hovered.id.title())
            self._active_app = app_name
            ts = time.strftime("%H:%M:%S")
            entry = (f"{app_name} opened", ts)
            self._app_log.insert(0, entry)
            if len(self._app_log) > 8:
                self._app_log.pop()

        if right_click_fired and hovered:
            from src.os.icons import APP_LABELS
            app_name = APP_LABELS.get(hovered.id, hovered.id.title())
            ts = time.strftime("%H:%M:%S")
            entry = (f"{app_name} [right-click]", ts)
            self._app_log.insert(0, entry)
            if len(self._app_log) > 8:
                self._app_log.pop()

        self._draw_title_bar(canvas, w)
        self._draw_taskbar(canvas, w, h)

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return canvas_rgb, list(self._app_log)

    def _draw_background(self, w: int, h: int) -> np.ndarray:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        bg = (26, 26, 46)
        canvas[:] = bg
        self._draw_grid_pattern(canvas, w, h)
        return canvas

    def _draw_grid_pattern(self, canvas: np.ndarray, w: int, h: int):
        grid_color = (36, 36, 60)
        for gx in range(0, w, 40):
            cv2.line(canvas, (gx, TITLE_H), (gx, h - TASKBAR_H), grid_color, 1, cv2.LINE_AA)
        for gy in range(TITLE_H, h - TASKBAR_H, 40):
            cv2.line(canvas, (0, gy), (w, gy), grid_color, 1, cv2.LINE_AA)

    def _draw_icons(self, canvas: np.ndarray, icons: list[IconDef], w: int, h: int,
                    hovered: IconDef | None):
        for icon in icons:
            ix, iy = self._icon_positions.get(icon.id, (icon.x, icon.y))

            is_hovered = hovered is not None and hovered.id == icon.id
            is_dragged = self._dragged_icon_id == icon.id

            if is_hovered:
                overlay = canvas.copy()
                cv2.rectangle(overlay, (ix - 4, iy - 4), (ix + icon.width + 4, iy + icon.height + 20),
                              (80, 150, 255), -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

            if is_dragged:
                overlay = canvas.copy()
                cv2.rectangle(overlay, (ix - 4, iy - 4), (ix + icon.width + 4, iy + icon.height + 20),
                              (100, 255, 160), -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

            draw_icon_shape(canvas, ix, iy, icon.width, icon.height, icon.shape, icon.color)

            label_y = iy + icon.height + 14
            if label_y + 12 < h - TASKBAR_H:
                cv2.putText(canvas, icon.label.split()[0], (ix - 4, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 230), 1, cv2.LINE_AA)

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
            cv2.putText(canvas, self._active_app, (36, y0 + TASKBAR_H // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 255), 1, cv2.LINE_AA)

        clock_str = time.strftime("%H:%M")
        cv2.putText(canvas, clock_str, (w - 50, y0 + TASKBAR_H // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 195, 220), 1, cv2.LINE_AA)

    def get_app_log(self) -> list[tuple[str, str]]:
        return list(self._app_log)