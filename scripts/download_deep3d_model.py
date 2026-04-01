#!/usr/bin/env python3
"""Download the default Deep3D CPU weights into Deep3D/export/deep3d_v1.0_640x360_cpu.pt.

Run from repo root:
  python3 scripts/download_deep3d_model.py

Requires: pip install gdown  (included in backend/requirements.txt)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Same default as backend/app.py — HypoX64 Google Drive (Deep3D pretrained export folder)
DEFAULT_FILENAME = "deep3d_v1.0_640x360_cpu.pt"
DEFAULT_GDRIVE_FILE_ID = "1oUB4vHqcgwXb7hMpSLtB2pxdNzY3kq7_"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Deep3D pretrained model weights.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Destination file (default: Deep3D/export/{DEFAULT_FILENAME})",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    target = args.output
    if target is None:
        target = root / "Deep3D" / "export" / DEFAULT_FILENAME
    else:
        target = target.resolve()

    try:
        import gdown
    except ImportError:
        print("Missing dependency: pip install gdown", file=sys.stderr)
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"Already present ({target.stat().st_size // 1_000_000} MB): {target}")
        return 0

    url = f"https://drive.google.com/uc?id={DEFAULT_GDRIVE_FILE_ID}"
    print(f"Downloading {DEFAULT_FILENAME} …")
    gdown.download(url, str(target), quiet=False)

    if not target.exists() or target.stat().st_size < 1_000_000:
        print("Download failed or file too small.", file=sys.stderr)
        if target.exists():
            target.unlink(missing_ok=True)
        return 1

    print(f"Saved: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
