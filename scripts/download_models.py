#!/usr/bin/env python3
"""
Download MediaPipe model files for HandMouse.

Run this once before first launch:
    python scripts/download_models.py

Models will be saved to models/gesture_recognizer.task
"""

import urllib.request
import hashlib
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-assets/gesture_recognizer.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
}


def download_with_progress(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name}...")
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 8192
        with open(dest, "wb") as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                f.write(buffer)
                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    print(f"\r  {percent:.1f}% ({downloaded // 1024}KB / {total_size // 1024}KB)", end="", flush=True)
    print(f"\n  Saved to {dest}")


def main():
    print("HandMouse — MediaPipe Model Downloader")
    print("=" * 40)

    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            size = dest.stat().st_size
            print(f"{filename} already exists ({size // 1024}KB) — skipping")
        else:
            try:
                download_with_progress(url, dest)
            except Exception as e:
                print(f"\n  ERROR: Failed to download {filename}: {e}")
                if dest.exists():
                    dest.unlink()
                sys.exit(1)

    print("\nAll models ready. Run 'bash run.sh' to start HandMouse.")


if __name__ == "__main__":
    main()