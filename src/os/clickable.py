from dataclasses import dataclass
from src.os.icons import IconDef, get_icon_grid


@dataclass
class IconEvent:
    icon_id: str
    action: str


def hit_test(cursor_x_norm: float, cursor_y_norm: float, icons: list[IconDef], w: int, h: int) -> IconDef | None:
    cx = int(cursor_x_norm * w)
    cy = int(cursor_y_norm * h)
    for icon in icons:
        if icon.x <= cx <= icon.x + icon.width and icon.y <= cy <= icon.y + icon.height:
            return icon
    return None


def get_icon_at(hands, w, h):
    pass