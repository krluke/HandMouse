# HandMouse

Real-time hand tracking and gesture recognition powered by MediaPipe. Displays a clean skeleton overlay on a black background with gesture classification in a sidebar — no camera feed clutter.

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-NixOS%20%7C%20Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)

## Features

- **Real-time hand tracking** at 50+ FPS (skeleton-only display mode)
- **21-landmark hand skeleton** with finger state detection
- **3-tier gesture classification**: MediaPipe → local rules → NVIDIA NIM LLM (optional)
- **Cross-platform**: NixOS, Linux, macOS, Windows (WSL)
- **Wayland native** with PySide6 GUI
- **Switchable display modes** — skeleton-only (black canvas) or camera + skeleton overlay (dropdown in sidebar)
- **Sidebar info panel**: gesture label, confidence, source, finger states, hand position

## Requirements

- Python 3.13+
- Webcam
- NVIDIA API key (optional — for NIM LLM gesture classification)

## Quick Start

### 1. Clone and setup environment

```bash
git clone https://github.com/krluke/HandMouse.git
cd HandMouse
```

### 2. Create Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS/NixOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Download MediaPipe models

```bash
python scripts/download_models.py
```

### 4. Run

```bash
bash run.sh
```

Without NIM (default — fast, local only):
```bash
bash run.sh
```

With NVIDIA NIM LLM gesture classification:
```bash
bash run.sh --nim
```

With custom config:
```bash
bash run.sh --config path/to/settings.yaml
```

## Project Structure

```
HandMouse/
├── main.py              # Entry point — Qt event loop, no threads
├── run.sh               # Cross-platform launcher (NixOS/Linux/macOS)
├── requirements.txt     # Python dependencies
├── config/
│   └── settings.yaml    # Camera, MediaPipe, NIM, display settings
├── models/              # MediaPipe .task files (downloaded via script)
├── scripts/
│   └── download_models.py  # Downloads MediaPipe model files
└── src/
    ├── camera.py        # OpenCV camera capture
    ├── hand_tracker.py  # MediaPipe GestureRecognizer (landmarks + gestures)
    ├── gesture_llm.py   # NVIDIA NIM Llama 3.2 Vision client
    ├── pipeline.py      # 3-tier gesture orchestration
    ├── renderer.py      # Skeleton-only renderer (black canvas)
    ├── gui.py           # PySide6 window, sidebar, video label
    └── utils.py         # Config loader (NIM_API_KEY env var support)
```

## Architecture

```
Camera frame (640x360)
       │
       ▼
MediaPipe GestureRecognizer VIDEO mode
       │
       ├── hand_landmarks → (21 coords × 3 floats per hand)
       ├── handedness → "Left" / "Right"
       ├── gestures → "Open_Palm", "Pointing_Up", ...
       │
       ▼
Pipeline — 3 classification tiers
       │
       ├─ Tier 1: MediaPipe gesture (from GestureRecognizer)
       ├─ Tier 2: Local rules (finger state → gesture name)
       └─ Tier 3: NIM LLM (on pose change, debounced) ← optional
       │
       ▼
Skeleton Renderer
       │
       ├─ Skeleton Only:  black canvas + bone overlay
       ├─ Camera + Skeleton: real camera feed + bone overlay
       ├─ Bone connections (white, fingertip segments cyan)
       ├─ Landmark dots (wrist=orange, fingertips=green, other=gray)
       └─ FPS counter
       │
       ▼
PySide6 GUI
       │
       ├─ Video panel: skeleton or camera+skeleton (switchable via dropdown)
       └─ Sidebar: gesture, confidence, source, finger states, position, display mode selector
```

## Configuration

Edit `config/settings.yaml`:

| Setting | Default | Description |
|---|---|---|
| `camera.device_index` | `0` | Webcam index |
| `camera.width` / `height` | `640 / 360` | Frame resolution |
| `mediapipe.min_detection_confidence` | `0.8` | Hand detection threshold |
| `mediapipe.min_tracking_confidence` | `0.6` | Landmark tracking threshold |
| `display.show_landmarks` | `true` | Show landmark dots |
| `display.show_connections` | `true` | Show bone connections |
| `display.show_fps` | `true` | Show FPS counter |

### NVIDIA NIM (optional)

Get your API key at [build.nvidia.com](https://build.nvidia.com/).

Set it via environment variable (recommended):
```bash
export NIM_API_KEY="nvapi-..."
bash run.sh --nim
```

Or edit `config/settings.yaml` directly (not recommended for shared machines).

## Performance

| Configuration | FPS | Notes |
|---|---|---|
| Skeleton-only, 640×360, no NIM | **50+ FPS** | Main mode |
| With NIM enabled | ~1 FPS for LLM calls | NIM calls debounced, don't affect display FPS |

The camera read is the main bottleneck (~30ms). The AI inference itself is ~8ms at 640×360.

## Platforms

| Platform | Notes |
|---|---|
| **NixOS** | Full support — `run.sh` auto-detects and sets Nix store paths |
| **Linux (non-NixOS)** | Full support — uses system library paths |
| **macOS** | Full support — `QT_QPA_PLATFORM=cocoa` |
| **Windows** | WSL recommended; native Windows: run `.venv\Scripts\python.exe main.py` directly |

## License

MIT — see [LICENSE](LICENSE)