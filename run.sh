#!/usr/bin/env bash
# run.sh — Cross-platform launcher for HandTracker
#
# Detects OS automatically and sets the right library paths:
#   - NixOS:     system-path dynamic (changes on upgrade), other libs hardcoded (stable)
#   - Linux:     system library paths
#   - macOS:     Homebrew or system paths, QT_QPA_PLATFORM=cocoa
#   - Windows:   see WINDOWS section below
#
# Usage:
#   bash run.sh                    # NIM disabled by default
#   bash run.sh --nim              # enable NIM LLM
#   bash run.sh --config custom.yaml
#
# ============================================================================
# WINDOWS (PowerShell or Git Bash / WSL)
#   On native Windows, run the venv Python directly:
#
#   .\venv\Scripts\python.exe main.py
#
#   Or on WSL2:
#   bash run.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

# ---- OS Detection ----
detect_os() {
    if [ -d /nix/store ] && grep -q "NixOS" /etc/os-release 2>/dev/null; then
        echo "nixos"
    elif [ "$(uname)" = "Darwin" ]; then
        echo "macos"
    elif [ "$(uname)" = "Linux" ]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

OS_TYPE="$(detect_os)"

# ---- Qt / Display Settings ----
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:=wayland}"
unset QT_STYLE_OVERRIDE QT_QPA_PLATFORMTHEME

# ---- OS-Specific Library Paths ----
if [ "$OS_TYPE" = "nixos" ]; then
    # ===== NIXOS =====
    # Find system-path: check /run symlinks first, then scan /nix/store for one with libGL
    SYSTEM_PATH_DIR=""
    for sp in /run/current-system/system-path /run/system-path; do
        if [ -e "$sp" ]; then
            SYSTEM_PATH_DIR="$sp"
            break
        fi
    done
    if [ -z "$SYSTEM_PATH_DIR" ] || [ ! -d "$SYSTEM_PATH_DIR" ]; then
        for d in /nix/store/*-system-path; do
            if [ -f "$d/lib/libGL.so.1" ]; then
                SYSTEM_PATH_DIR="$d/lib"
                break
            fi
        done
    fi

    # Versioned package paths (stable across nixos-version upgrades — same package hashes)
    NIX_LIBS=(
        "$SYSTEM_PATH_DIR"
        "/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib"
        "/nix/store/fc1g44pg3i10wfzh3gb4m54pfgclsn76-libxcb-1.17.0/lib"
        "/nix/store/zcmsivndca5wmam9nwnbjrm0zkgykwfz-glib-2.86.3/lib"
        "/nix/store/ixhlv41i2wpl84xgjcks061dz4yssbg3-zlib-1.3.2/lib"
        "/nix/store/yanmwp5f435ing2nbhwa4v0gdmpl2an1-dbus-1.16.2-lib/lib"
        "/nix/store/0c0xdj7xpilqfy2p33l1jm407f01652w-libxkbcommon-1.13.1/lib"
        "/nix/store/ysbyz6zabjcg078ssp4l58mhgbr57pbz-wayland-1.24.0/lib"
        "/nix/store/bg6ms0vw071g1fdbx2my6bbzsk62p6vd-fontconfig-2.17.1-lib/lib"
        "/nix/store/zr22ggqbv79yv4y4wv06r4grla9h59yx-freetype-2.14.2/lib"
        "/nix/store/k0rqiflg1vkn1kj96br5pfxj40p3srz4-zstd-1.5.7/lib"
    )

    LIB_PATH=""
    for dir in "${NIX_LIBS[@]}"; do
        if [ -d "$dir" ]; then
            LIB_PATH="${LIB_PATH:+$LIB_PATH:}$dir"
        fi
    done
    export LD_LIBRARY_PATH="${LIB_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    export QT_PLUGIN_PATH="$SCRIPT_DIR/.venv/lib/python3.13/site-packages/PySide6/Qt/plugins${QT_PLUGIN_PATH:+:$QT_PLUGIN_PATH}"
    export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-1}"

elif [ "$OS_TYPE" = "macos" ]; then
    # ===== macOS =====
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:=cocoa}"
    unset QT_PLUGIN_PATH

    HOMEBREW_PATHS=(
        "/opt/homebrew/lib"
        "/usr/local/lib"
        "$HOME/Homebrew/lib"
    )
    LIB_PATH=""
    for dir in "${HOMEBREW_PATHS[@]}"; do
        if [ -d "$dir" ]; then
            LIB_PATH="${LIB_PATH:+$LIB_PATH:}$dir"
        fi
    done
    export LD_LIBRARY_PATH="${LIB_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

elif [ "$OS_TYPE" = "linux" ]; then
    # ===== Linux (non-NixOS) =====
    if [ -n "${WAYLAND_DISPLAY:-}" ]; then
        export QT_QPA_PLATFORM="wayland"
    elif [ -n "${DISPLAY:-}" ]; then
        export QT_QPA_PLATFORM="xcb"
    else
        export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland}"
    fi
    unset QT_PLUGIN_PATH

    LINUX_LIB_PATHS="/usr/lib:/usr/local/lib:/usr/lib64"
    if [ -d "/opt/vc/lib" ]; then
        LINUX_LIB_PATHS="/opt/vc/lib:$LINUX_LIB_PATHS"
    fi
    export LD_LIBRARY_PATH="${LINUX_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

else
    # ===== Unknown / Windows WSL fallback =====
    unset QT_PLUGIN_PATH
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland}"
fi

# ---- Launch the app ----
exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" "$@"