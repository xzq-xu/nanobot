#!/usr/bin/env python3
"""
Build the nanobot-gateway standalone binary via PyInstaller.

Usage:
    python scripts/build-gateway.py            # build for current platform
    python scripts/build-gateway.py --verify   # build + health-check the binary

The output lands in dist/nanobot-gateway-{platform}-{arch}.
"""

import argparse
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "nanobot-gateway.spec"

_OS = {"Darwin": "darwin", "Linux": "linux", "Windows": "win"}
_ARCH = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "AMD64": "x64"}


def binary_name() -> str:
    plat = _OS.get(platform.system(), platform.system().lower())
    arch = _ARCH.get(platform.machine(), platform.machine())
    ext = ".exe" if platform.system() == "Windows" else ""
    return f"nanobot-gateway-{plat}-{arch}{ext}"


def build() -> Path:
    print(f"[build] spec: {SPEC}")
    print(f"[build] target: {binary_name()}")

    subprocess.check_call(
        [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", str(SPEC)],
        cwd=str(ROOT),
    )

    out = ROOT / "dist" / binary_name()
    if not out.exists():
        print(f"[build] ERROR: expected output not found at {out}")
        sys.exit(1)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"[build] OK — {out}  ({size_mb:.1f} MB)")
    return out


def verify(binary: Path, port: int = 19999) -> None:
    """Start the binary and hit /health to confirm it works."""
    print(f"[verify] starting {binary.name} on port {port}...")
    proc = subprocess.Popen(
        [str(binary), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    ok = False
    try:
        for attempt in range(20):
            time.sleep(3)
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
                if resp.status == 200:
                    print(f"[verify] health check passed (attempt {attempt + 1})")
                    ok = True
                    break
            except Exception:
                    print(f"[verify] waiting... ({attempt + 1}/20)")
        if not ok:
            print("[verify] FAILED — gateway did not become healthy in 60s")
            sys.exit(1)
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)

    print("[verify] OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nanobot-gateway binary")
    parser.add_argument("--verify", action="store_true", help="Run health check after build")
    parser.add_argument("--clean", action="store_true", help="Remove build/dist dirs first")
    args = parser.parse_args()

    if args.clean:
        for d in ["build", "dist"]:
            p = ROOT / d
            if p.exists():
                print(f"[clean] removing {p}")
                shutil.rmtree(p)

    binary = build()

    if args.verify:
        verify(binary)

    print(f"\nDone. Binary: {binary}")


if __name__ == "__main__":
    main()
