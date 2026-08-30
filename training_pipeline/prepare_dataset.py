#!/usr/bin/env python3
"""
Extract synchronized frame pairs from rectified left video (source) and disparity /
depth-map video (target) for Deep3D-style training.

Outputs under training_pipeline/dataset/ as frame_XXXX_src.png and frame_XXXX_disp.png
(4-digit index starting at 0001). Frames are resized to 640x360. Disparity is passed
through scale_for_shrew() (placeholder IPD ratio 12/63, tunable).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DEEP3D = _REPO / "Deep3D"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_DEEP3D) not in sys.path:
    sys.path.insert(0, str(_DEEP3D))

from data import impro  # noqa: E402
from utils import util  # noqa: E402

# Deep3D export model default resolution (width x height)
MODEL_W = 640
MODEL_H = 360

# Placeholder: tree shrew ~12 mm IPD vs ~63 mm human reference (tune as needed)
SHREW_IPD_MM = 12.0
REF_IPD_MM = 63.0
SHREW_SCALE = SHREW_IPD_MM / REF_IPD_MM  # 12/63


def scale_for_shrew(disparity_map: np.ndarray) -> np.ndarray:
    """
    Adjust disparity / depth-proxy values for a 12 mm IPD subject vs 63 mm reference.

    This is a linear placeholder: scaled = disparity * (12/63). Replace SHREW_SCALE
    or add calibration-aware logic when you have measured baselines.

    Accepts uint8/float single- or multi-channel (uses first channel if multi).
    Returns uint8 in [0, 255] suitable for saving as grayscale PNG.
    """
    if disparity_map.ndim == 3:
        d = disparity_map[:, :, 0].astype(np.float64)
    else:
        d = disparity_map.astype(np.float64)
    scaled = d * SHREW_SCALE
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def resize_pair(
    bgr_src: np.ndarray, disp_gray: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resize source BGR and disparity grayscale to MODEL_W x MODEL_H."""
    size = (MODEL_W, MODEL_H)
    src_r = cv2.resize(bgr_src, size, interpolation=cv2.INTER_AREA)
    disp_r = cv2.resize(disp_gray, size, interpolation=cv2.INTER_AREA)
    return src_r, disp_r


def extract_pairs(
    left_video: Path,
    disparity_video: Path,
    out_dir: Path,
) -> int:
    cap_l = cv2.VideoCapture(str(left_video))
    cap_d = cv2.VideoCapture(str(disparity_video))
    if not cap_l.isOpened() or not cap_d.isOpened():
        raise RuntimeError(
            f"Could not open videos: {left_video} | {disparity_video}"
        )

    util.makedirs(str(out_dir))

    idx = 0
    while True:
        ret_l, fr_l = cap_l.read()
        ret_d, fr_d = cap_d.read()
        if not ret_l or not ret_d or fr_l is None or fr_d is None:
            break

        if fr_d.ndim == 3:
            disp_gray = cv2.cvtColor(fr_d, cv2.COLOR_BGR2GRAY)
        else:
            disp_gray = fr_d

        disp_scaled = scale_for_shrew(disp_gray)
        fr_l_r, disp_r = resize_pair(fr_l, disp_scaled)

        # frame_0001_src.png, frame_0001_disp.png (1-based 4-digit index)
        n = idx + 1
        stem = f"frame_{n:04d}"
        path_src = out_dir / f"{stem}_src.png"
        path_disp = out_dir / f"{stem}_disp.png"

        impro.imwrite(str(path_src), fr_l_r, use_thread=False)
        # Single-channel disparity PNG
        cv2.imwrite(str(path_disp), disp_r)

        idx += 1
        if idx % 200 == 0:
            logging.info("Extracted %d pairs …", idx)

    cap_l.release()
    cap_d.release()

    logging.info("Saved %d pairs to %s (640x360, shrew scale=%.6f)", idx, out_dir, SHREW_SCALE)
    return idx


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Extract src/disparity frame pairs for training (640x360)."
    )
    p.add_argument(
        "--left-video",
        type=Path,
        default=root / "training_pipeline" / "rectified" / "rect_L.avi",
        help="Rectified left video (source).",
    )
    p.add_argument(
        "--disparity-video",
        type=Path,
        default=root / "training_pipeline" / "disparity" / "depth_map.avi",
        help="Disparity / depth-map video (target), e.g. depth_map.avi.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "training_pipeline" / "dataset",
        help="Output directory for frame_XXXX_src.png and frame_XXXX_disp.png.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)
    try:
        n = extract_pairs(
            args.left_video.resolve(),
            args.disparity_video.resolve(),
            args.out_dir.resolve(),
        )
        if n == 0:
            logging.error("No frames extracted — check video paths and codecs.")
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
