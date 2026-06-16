from dataclasses import dataclass, field


@dataclass
class MouseState:
    cursor_x: float = 0.5
    cursor_y: float = 0.5
    cursor_visible: bool = False
    left_click: bool = False
    right_click: bool = False
    scroll_dy: float = 0.0
    is_picking: bool = False
    is_dragging: bool = False
    confidence: float = 0.0

    def clear_events(self):
        self.left_click = False
        self.right_click = False
        self.scroll_dy = 0.0


@dataclass
class GestureAction:
    type: str = "none"
    detail: str = ""