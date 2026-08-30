#!/usr/bin/env python3
"""
Rectify dual-camera AVI/MOV using stereo_calib.npz (macOS-friendly).

**Production mode (default):** getOptimalNewCameraMatrix + stereoRectify(alpha=0), 1280x720 MJPG.

**Diagnostic mode (`--diagnostic`):** raw M1/M2, stereoRectify **alpha=1** (retain all pixels,
visible black borders) — use when one eye is black to see if content is shifted off-canvas.
Maps use CV_16SC2; output size matches input video resolution.
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
from training_pipeline.calib_io import StereoCalibration, rms_log_lines  # noqa: E402

# Fixed output / training resolution (must match input AVI from your capture rig)
OUT_W = 1280
OUT_H = 720


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_intrinsics_ex_rt(calib_path: Path) -> tuple[np.ndarray, ...]:
    """M1,d1,M2,d2,R,T from NPZ (either M1 or mtxL naming)."""
    data = np.load(calib_path, allow_pickle=True)
    keys = set(data.files)
    if {"M1", "d1", "M2", "d2", "R", "T"}.issubset(keys):
        return data["M1"], data["d1"], data["M2"], data["d2"], data["R"], data["T"]
    if {"mtxL", "distL", "mtxR", "distR", "R", "T"}.issubset(keys):
        return data["mtxL"], data["distL"], data["mtxR"], data["distR"], data["R"], data["T"]
    raise KeyError(
        f"{calib_path} must contain M1,d1,M2,d2,R,T or mtxL,distL,mtxR,distR,R,T"
    )


def _log_calib_rms(calib_path: Path) -> None:
    try:
        cal = StereoCalibration.from_npz(calib_path)
        for line in rms_log_lines(cal):
            logging.info(line)
    except Exception as exc:  # noqa: BLE001
        logging.debug("Could not load extended calib metadata: %s", exc)


def build_rectify_maps(
    M1: np.ndarray,
    d1: np.ndarray,
    M2: np.ndarray,
    d2: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    in_w: int,
    in_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimal new camera matrices (alpha=0) for both eyes, then stereoRectify with the
    same output canvas (OUT_W x OUT_H), then undistort+rectify maps at that size.
    """
    in_size = (in_w, in_h)
    out_size = (OUT_W, OUT_H)

    # alpha=0: crop to valid pixels only (avoids huge black margins that break writers on macOS)
    new_m1, _roi1 = cv2.getOptimalNewCameraMatrix(M1, d1, in_size, alpha=0, newImgSize=out_size)
    new_m2, _roi2 = cv2.getOptimalNewCameraMatrix(M2, d2, in_size, alpha=0, newImgSize=out_size)

    R1, R2, P1, P2, _q, _roi_l, _roi_r = cv2.stereoRectify(
        new_m1,
        d1,
        new_m2,
        d2,
        in_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
        newImageSize=out_size,
    )

    map_l1, map_l2 = cv2.initUndistortRectifyMap(
        new_m1, d1, R1, P1, out_size, cv2.CV_32FC1
    )
    map_r1, map_r2 = cv2.initUndistortRectifyMap(
        new_m2, d2, R2, P2, out_size, cv2.CV_32FC1
    )
    return map_l1, map_l2, map_r1, map_r2


def rectify_diagnostic(
    left_path: Path,
    right_path: Path,
    calib_path: Path,
    out_l: Path,
    out_r: Path,
    *,
    preview: bool,
) -> None:
    """
    Alpha=1 rectification: no getOptimalNewCameraMatrix; shows full rectified image including
    borders so you can see if asymmetric calibration pushes one view off-screen.
    """
    if not calib_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    _log_calib_rms(calib_path)
    M1, d1, M2, d2, R, T = _load_intrinsics_ex_rt(calib_path)

    cap_l = cv2.VideoCapture(str(left_path))
    cap_r = cv2.VideoCapture(str(right_path))
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError(f"Could not open: {left_path} | {right_path}")

    ret_l, f_l = cap_l.read()
    ret_r, f_r = cap_r.read()
    if not ret_l or not ret_r or f_l is None or f_r is None:
        cap_l.release()
        cap_r.release()
        raise RuntimeError("Could not read first frame.")

    in_h, in_w = f_l.shape[:2]
    if f_r.shape[:2] != (in_h, in_w):
        cap_l.release()
        cap_r.release()
        raise RuntimeError("Left/right frame size mismatch.")

    # CAP_PROP can be 0 before decode; prefer decoded frame size
    w_prop = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h_prop = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w_prop and h_prop and (w_prop, h_prop) != (in_w, in_h):
        logging.warning(
            "CAP_PROP size %dx%d != first frame %dx%d; using first frame.",
            w_prop,
            h_prop,
            in_w,
            in_h,
        )

    fps = float(cap_l.get(cv2.CAP_PROP_FPS) or 0.0) or float(cap_r.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1e-3:
        fps = 20.0
        logging.warning("FPS not reported; using 20 (diagnostic default).")

    target_size = (in_w, in_h)

    # Change the alpha and flags here
    # alpha=1 means "Show me everything, even if there are black borders"
    R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(
        M1,
        d1,
        M2,
        d2,
        target_size,
        R,
        T,
        alpha=1,
        newImageSize=target_size,
        flags=cv2.CALIB_ZERO_DISPARITY,  # helps keep the images centered
    )
    logging.debug("stereoRectify valid ROI left=%s right=%s", validRoi1, validRoi2)

    map_l1, map_l2 = cv2.initUndistortRectifyMap(
        M1, d1, R1, P1, target_size, cv2.CV_16SC2
    )
    map_r1, map_r2 = cv2.initUndistortRectifyMap(
        M2, d2, R2, P2, target_size, cv2.CV_16SC2
    )

    util.makedirs(str(out_l.parent))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    size_wh = target_size
    out_path_l = str(out_l.resolve())
    out_path_r = str(out_r.resolve())
    writer_l = cv2.VideoWriter(out_path_l, fourcc, fps, size_wh, True)
    writer_r = cv2.VideoWriter(out_path_r, fourcc, fps, size_wh, True)
    if not writer_l.isOpened() or not writer_r.isOpened():
        writer_l.release()
        writer_r.release()
        cap_l.release()
        cap_r.release()
        raise RuntimeError("Could not open MJPG VideoWriters (diagnostic).")

    logging.info(
        "Diagnostic rectification: alpha=1, size %dx%d, MJPG | %s | %s",
        in_w,
        in_h,
        out_path_l,
        out_path_r,
    )
    print("Running Diagnostic Rectification (alpha=1 — borders visible; press 'q' in preview to stop)…")

    cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)

    win = "Diagnostic (Are the images visible but surrounded by black?)"
    if preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    count = 0
    try:
        while True:
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r or frame_l is None or frame_r is None:
                break

            rect_l = cv2.remap(frame_l, map_r1, map_r2, cv2.INTER_LINEAR)
            rect_r = cv2.remap(frame_r, map_r1, map_r2, cv2.INTER_LINEAR)

            writer_l.write(rect_l)
            writer_r.write(rect_r)

            if preview:
                vis = np.hstack((rect_l, rect_r))
                cv2.imshow(win, vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logging.info("Quit preview (q).")
                    break
            count += 1
    finally:
        if preview:
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                cv2.destroyAllWindows()
        writer_l.release()
        writer_r.release()
        cap_l.release()
        cap_r.release()

    logging.info("Diagnostic: wrote %d frame pairs.", count)


def rectify_video(
    left_path: Path,
    right_path: Path,
    calib_path: Path,
    out_l: Path,
    out_r: Path,
    *,
    debug_window: bool,
) -> None:
    if not calib_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    _log_calib_rms(calib_path)
    M1, d1, M2, d2, R, T = _load_intrinsics_ex_rt(calib_path)

    cap_l = cv2.VideoCapture(str(left_path))
    cap_r = cv2.VideoCapture(str(right_path))
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError(f"Could not open: {left_path} or {right_path}")

    ret_l, frame_l0 = cap_l.read()
    ret_r, frame_r0 = cap_r.read()
    if not ret_l or not ret_r or frame_l0 is None or frame_r0 is None:
        cap_l.release()
        cap_r.release()
        raise RuntimeError("Could not read the first frame from one or both videos.")

    in_h, in_w = frame_l0.shape[:2]
    h2, w2 = frame_r0.shape[:2]
    if (in_h, in_w) != (h2, w2):
        cap_l.release()
        cap_r.release()
        raise RuntimeError(f"Frame size mismatch: left {in_w}x{in_h} vs right {w2}x{h2}")

    if (in_w, in_h) != (OUT_W, OUT_H):
        logging.warning(
            "Input frame size is %dx%d; pipeline assumes %dx%d. "
            "Calibration/maps assume input matches this resolution.",
            in_w,
            in_h,
            OUT_W,
            OUT_H,
        )

    fps = float(cap_l.get(cv2.CAP_PROP_FPS) or 0.0) or float(cap_r.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1e-3:
        fps = 30.0
        logging.warning("FPS not reported by capture; using 30")

    map_l1, map_l2, map_r1, map_r2 = build_rectify_maps(
        M1, d1, M2, d2, R, T, in_w, in_h
    )

    util.makedirs(str(out_l.parent))

    for p, label in ((out_l, "left"), (out_r, "right")):
        if p.suffix.lower() != ".avi":
            logging.warning("Output %s path %s is not .avi — MJPG is most reliable with .avi on macOS.", label, p)

    # Force-Match: single canonical size for writers + every written frame (width, height)
    target_size = (1280, 720)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_path_l = str(out_l.resolve())
    out_path_r = str(out_r.resolve())
    writer_l = cv2.VideoWriter(out_path_l, fourcc, fps, target_size, True)
    writer_r = cv2.VideoWriter(out_path_r, fourcc, fps, target_size, True)

    if not writer_l.isOpened() or not writer_r.isOpened():
        writer_l.release()
        writer_r.release()
        cap_l.release()
        cap_r.release()
        raise RuntimeError(
            "Could not open MJPG VideoWriter for left and/or right. "
            f"Paths: {out_path_l}, {out_path_r}"
        )

    logging.info(
        "MJPG AVI %dx%d @ %.3f fps (forced size) | left: %s | right: %s",
        OUT_W,
        OUT_H,
        fps,
        out_path_l,
        out_path_r,
    )

    cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if debug_window:
        cv2.namedWindow("Debug Right Eye", cv2.WINDOW_NORMAL)

    count = 0
    try:
        while True:
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r or frame_l is None or frame_r is None:
                break

            rect_l = cv2.remap(
                frame_l, map_r1, map_r2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
            rect_r = cv2.remap(
                frame_r, map_r1, map_r2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )

            rect_l = cv2.resize(rect_l, target_size, interpolation=cv2.INTER_LINEAR)
            rect_r = cv2.resize(rect_r, target_size, interpolation=cv2.INTER_LINEAR)

            print(f"Writing Frame {count}: L_shape={rect_l.shape}, R_shape={rect_r.shape}")

            mean_l = float(np.mean(rect_l))
            mean_r = float(np.mean(rect_r))
            if mean_l == 0:
                print(
                    f"ERROR: Left eye frame is black (mean pixel value is 0). Frame {count}."
                )
            if mean_r == 0:
                print(
                    f"ERROR: Right eye frame is black (mean pixel value is 0). Frame {count}."
                )

            writer_l.write(rect_l)
            writer_r.write(rect_r)

            if debug_window:
                cv2.imshow("Debug Right Eye", rect_r)
                cv2.waitKey(1)

            count += 1
    finally:
        if debug_window:
            try:
                cv2.destroyWindow("Debug Right Eye")
            except cv2.error:
                pass
        writer_l.release()
        writer_r.release()
        cap_l.release()
        cap_r.release()

    logging.info("Processed %d frame pairs.", count)
    logging.info("Outputs: %s | %s", out_l, out_r)


def _default_stereo_inputs(root: Path) -> tuple[Path, Path]:
    """First existing (left, right) pair from common repo locations."""
    candidates: list[tuple[Path, Path]] = [
        (root / "cam_L_32mm.avi", root / "cam_R_32mm.avi"),
        (
            root / "calibration" / "videos" / "cam_L.avi",
            root / "calibration" / "videos" / "cam_R.avi",
        ),
        (
            root / "calibration" / "videos 2" / "cam_L.avi",
            root / "calibration" / "videos 2" / "cam_R.avi",
        ),
    ]
    for left, right in candidates:
        if left.is_file() and right.is_file():
            return left, right
    raise FileNotFoundError(
        "No default left/right AVI pair found. Pass --left and --right explicitly. "
        f"Tried: {[str(a) for a, b in candidates]}"
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Rectify stereo: production (1280x720, optimal K) or --diagnostic (alpha=1)."
    )
    p.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "Use stereoRectify(alpha=1), no getOptimalNewCameraMatrix, CV_16SC2 maps, "
            "native video size — shows borders to debug black / shifted views."
        ),
    )
    p.add_argument(
        "--left",
        type=Path,
        default=None,
        help="Left camera video (default: auto-detect cam_L_32mm.avi or calibration/videos).",
    )
    p.add_argument(
        "--right",
        type=Path,
        default=None,
        help="Right camera video (default: auto-detect paired with --left).",
    )
    p.add_argument(
        "--calib",
        type=Path,
        default=root / "stereo_calib.npz",
        help="Calibration NPZ (default: repo root stereo_calib.npz).",
    )
    p.add_argument(
        "--out-left",
        type=Path,
        default=root / "training_pipeline" / "rectified" / "rect_L.avi",
        help="Output rectified left.",
    )
    p.add_argument(
        "--out-right",
        type=Path,
        default=root / "training_pipeline" / "rectified" / "rect_R.avi",
        help="Output rectified right.",
    )
    p.add_argument(
        "--no-debug-window",
        action="store_true",
        help="Disable cv2.imshow debug (for headless / SSH).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)
    root = Path(__file__).resolve().parents[1]
    try:
        left = args.left
        right = args.right
        if left is None and right is None:
            left, right = _default_stereo_inputs(root)
            logging.info("Using default inputs: %s | %s", left, right)
        elif left is None or right is None:
            logging.error("Pass both --left and --right, or omit both for auto-detect.")
            return 1

        left = Path(left).resolve()
        right = Path(right).resolve()

        if args.diagnostic:
            rectify_diagnostic(
                left,
                right,
                args.calib.resolve(),
                args.out_left.resolve(),
                args.out_right.resolve(),
                preview=not args.no_debug_window,
            )
        else:
            rectify_video(
                left,
                right,
                args.calib.resolve(),
                args.out_left.resolve(),
                args.out_right.resolve(),
                debug_window=not args.no_debug_window,
            )
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
