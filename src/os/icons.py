from dataclasses import dataclass

APP_LABELS = {
    "explorer": "📁  Explorer",
    "calculator": "🔢  Calculator",
    "music": "🎵  Music",
    "browser": "🌐  Browser",
    "camera": "📷  Camera",
    "settings": "⚙️  Settings",
}

APP_COLORS = {
    "explorer": (80, 180, 255),
    "calculator": (255, 200, 80),
    "music": (200, 80, 255),
    "browser": (80, 255, 180),
    "camera": (255, 120, 80),
    "settings": (180, 180, 180),
}

APP_SHAPES = {
    "explorer": "folder",
    "calculator": "calc",
    "music": "music",
    "browser": "globe",
    "camera": "camera",
    "settings": "gear",
}


@dataclass
class IconDef:
    id: str
    label: str
    color: tuple
    shape: str
    x: int
    y: int
    width: int = 60
    height: int = 60


def get_icon_grid(w: int, h: int) -> list[IconDef]:
    rows, cols = 2, 3
    gap_x = 24
    gap_y = 20
    icon_w = 60
    icon_h = 60
    start_x = (w - (cols * icon_w + (cols - 1) * gap_x)) // 2
    start_y = h // 2 - 30

    icons = []
    all_ids = list(APP_LABELS.keys())

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(all_ids):
                break
            app_id = all_ids[idx]
            x = start_x + c * (icon_w + gap_x)
            y = start_y + r * (icon_h + gap_y)
            icons.append(IconDef(
                id=app_id,
                label=APP_LABELS[app_id],
                color=APP_COLORS[app_id],
                shape=APP_SHAPES[app_id],
                x=x, y=y, width=icon_w, height=icon_h,
            ))

    return icons