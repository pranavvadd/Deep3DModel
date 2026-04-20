#!/usr/bin/env python3
"""
Disparity / depth-style video from rectified stereo pair (AVI/MOV).

Uses StereoSGBM + ximgproc DisparityWLSFilter (requires opencv-contrib), normalizes
to 8-bit grayscale, writes MJPG AVI and shows a live colormap preview.
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

from utils import util  # noqa: E402


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _require_ximgproc() -> None:
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError(
            "cv2.ximgproc is not available (needed for DisparityWLSFilter). "
            "Install OpenCV with contrib modules, e.g.: pip install opencv-contrib-python"
        )


def create_sgbm_small_baseline() -> cv2.StereoSGBM:
    """
    Default StereoSGBM tuned for a small (~12 mm) baseline: narrow disparity range.
    """
    block_size = 5
    p1 = 8 * 3 * block_size**2
    p2 = 32 * 3 * block_size**2
    mode = getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", cv2.STEREO_SGBM_MODE_SGBM)
    return cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=32,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=mode,
    )


def run_disparity_video(
    left_video: Path,
    right_video: Path,
    out_video: Path,
    *,
    preview: bool,
) -> int:
    _require_ximgproc()

    left_matcher = create_sgbm_small_baseline()
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(8000.0)
    wls.setSigmaColor(1.5)

    cap_l = cv2.VideoCapture(str(left_video))
    cap_r = cv2.VideoCapture(str(right_video))
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError(f"Could not open videos: {left_video} | {right_video}")

    fps = float(cap_l.get(cv2.CAP_PROP_FPS) or 0.0) or float(cap_r.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1e-3:
        fps = 30.0
        logging.warning("FPS not reported; using 30")

    ret_l, f_l = cap_l.read()
    ret_r, f_r = cap_r.read()
    if not ret_l or not ret_r or f_l is None or f_r is None:
        cap_l.release()
        cap_r.release()
        raise RuntimeError("Could not read first frame from one or both videos.")

    h, w = f_l.shape[:2]
    if f_r.shape[:2] != (h, w):
        cap_l.release()
        cap_r.release()
        raise RuntimeError(f"Frame size mismatch: {w}x{h} vs {f_r.shape[1]}x{f_r.shape[0]}")

    # WLS output matches left image size
    out_w, out_h = w, h

    util.makedirs(str(out_video.parent))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(out_video.resolve()), fourcc, fps, (out_w, out_h), True)
    if not writer.isOpened():
        cap_l.release()
        cap_r.release()
        raise RuntimeError(f"Could not open MJPG VideoWriter: {out_video}")

    if preview:
        cv2.namedWindow("Depth Heatmap", cv2.WINDOW_NORMAL)

    def process_frame(f_left: np.ndarray, f_right: np.ndarray) -> np.ndarray:
        g_l = cv2.cvtColor(f_left, cv2.COLOR_BGR2GRAY)
        g_r = cv2.cvtColor(f_right, cv2.COLOR_BGR2GRAY)
        dl = left_matcher.compute(g_l, g_r)
        dr = right_matcher.compute(g_r, g_l)
        filt = wls.filter(dl, f_left, dr, f_right)
        if filt.dtype != np.float32:
            filt = filt.astype(np.float32)
        # Visible grayscale depth: normalize to 0–255
        return cv2.normalize(filt, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)

    count = 0
    try:
        while True:
            ret_l, f_l = cap_l.read()
            ret_r, f_r = cap_r.read()
            if not ret_l or not ret_r or f_l is None or f_r is None:
                break
            vis = process_frame(f_l, f_r)
            writer.write(cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR))
            if preview:
                heat = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
                cv2.imshow("Depth Heatmap", heat)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logging.info("Quit requested (q).")
                    break
            count += 1
            if count % 100 == 0:
                logging.info("Processed %d frames", count)
    finally:
        if preview:
            try:
                cv2.destroyWindow("Depth Heatmap")
            except cv2.error:
                pass
        writer.release()
        cap_l.release()
        cap_r.release()

    logging.info("Wrote %d frames to %s (MJPG)", count, out_video)
    return 0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Rectified stereo → WLS-smoothed disparity video (MJPG AVI)."
    )
    p.add_argument(
        "--left-video",
        type=Path,
        default=root / "training_pipeline" / "rectified" / "rect_L.avi",
        help="Rectified left-eye video (default: training_pipeline/rectified/rect_L.avi).",
    )
    p.add_argument(
        "--right-video",
        type=Path,
        default=root / "training_pipeline" / "rectified" / "rect_R.avi",
        help="Rectified right-eye video (default: training_pipeline/rectified/rect_R.avi).",
    )
    p.add_argument(
        "--out-video",
        type=Path,
        default=root / "training_pipeline" / "disparity" / "depth_map.avi",
        help="Output MJPG depth visualization (default: training_pipeline/disparity/depth_map.avi).",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live Depth Heatmap window.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)
    try:
        return run_disparity_video(
            args.left_video.resolve(),
            args.right_video.resolve(),
            args.out_video.resolve(),
            preview=not args.no_preview,
        )
    except Exception as exc:  # noqa: BLE001
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
