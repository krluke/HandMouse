from dataclasses import dataclass
from src.os.icons import IconDef, get_icon_grid


@dataclass
class IconEvent:
    icon_id: str
    action: str


def hit_test(cursor_x_norm: float, cursor_y_norm: float,
             icon_positions: dict[str, tuple[int, int]],
             icons: list[IconDef], w: int, h: int) -> IconDef | None:
    cx = int(cursor_x_norm * w)
    cy = int(cursor_y_norm * h)
    for icon in icons:
        if icon.id not in icon_positions:
            continue
        ix, iy = icon_positions[icon.id]
        if ix <= cx <= ix + icon.width and iy <= cy <= iy + icon.height:
            return icon
    return None