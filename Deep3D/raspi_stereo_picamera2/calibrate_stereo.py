#!/usr/bin/env python3
"""
Stereo calibration from synced left/right image pairs (chessboard).

Uses rig JSON for board geometry and output paths. Re-run after changing cameras
or baseline; no code edits required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from rig_config import RigConfig, load_rig, save_rig_manifest


def _discover_pairs(pairs_dir: Path) -> List[Tuple[Path, Path]]:
    """Match *_<left_tag>.ext and *_<right_tag>.ext with same stem prefix."""
    left_tag, right_tag = "left", "right"
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    by_stem: dict[str, dict[str, Path]] = {}
    for pattern in exts:
        for p in pairs_dir.glob(pattern):
            name = p.stem.lower()
            for tag, key in ((left_tag, "L"), (right_tag, "R")):
                suffix = f"_{tag}"
                if name.endswith(suffix):
                    stem_key = name[: -len(suffix)]
                    by_stem.setdefault(stem_key, {})[key] = p
                    break
    pairs: List[Tuple[Path, Path]] = []
    for stem, sides in sorted(by_stem.items()):
        if "L" in sides and "R" in sides:
            pairs.append((sides["L"], sides["R"]))
    return pairs


def _object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= square_m
    return obj


def run_calibration(
    rig: RigConfig,
    pairs_dir: Path,
    alpha: float,
) -> int:
    pairs = _discover_pairs(pairs_dir.resolve())
    if len(pairs) < 5:
        print(
            f"Need at least 5 stereo pairs; found {len(pairs)} in {pairs_dir}. "
            "Name files like pair_000_left.jpg / pair_000_right.jpg",
            file=sys.stderr,
        )
        return 1

    cols = rig.chessboard_inner_cols
    rows = rig.chessboard_inner_rows
    square_m = rig.square_size_mm / 1000.0
    pattern_size = (cols, rows)
    obj_template = _object_points(cols, rows, square_m)

    obj_pts: List[np.ndarray] = []
    img_pts_l: List[np.ndarray] = []
    img_pts_r: List[np.ndarray] = []
    image_size: Tuple[int, int] | None = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        60,
        1e-4,
    )

    for pl, pr in pairs:
        gl = cv2.imread(str(pl), cv2.IMREAD_GRAYSCALE)
        gr = cv2.imread(str(pr), cv2.IMREAD_GRAYSCALE)
        if gl is None or gr is None:
            print(f"Skip unreadable: {pl} / {pr}", file=sys.stderr)
            continue
        if gl.shape != gr.shape:
            print(f"Skip size mismatch: {pl} vs {pr}", file=sys.stderr)
            continue
        h, w = gl.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            print(
                f"Skip inconsistent resolution: expected {image_size}, got {(w, h)}",
                file=sys.stderr,
            )
            continue

        ret_l, corners_l = cv2.findChessboardCorners(gl, pattern_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gr, pattern_size, None)
        if not ret_l or not ret_r:
            print(f"Skip (board not found): {pl.name} / {pr.name}", file=sys.stderr)
            continue

        corners_l = cv2.cornerSubPix(gl, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gr, corners_r, (11, 11), (-1, -1), criteria)

        obj_pts.append(obj_template)
        img_pts_l.append(corners_l)
        img_pts_r.append(corners_r)

    used = len(obj_pts)
    if used < 5:
        print(f"Need >=5 valid board detections; got {used}.", file=sys.stderr)
        return 1

    assert image_size is not None
    w, h = image_size

    _rl, M1, d1, _rvl, _tvl = cv2.calibrateCamera(
        obj_pts, img_pts_l, image_size, None, None, criteria=criteria, flags=0
    )
    _rr, M2, d2, _rvr, _tvr = cv2.calibrateCamera(
        obj_pts, img_pts_r, image_size, None, None, criteria=criteria, flags=0
    )
    rms, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts,
        img_pts_l,
        img_pts_r,
        M1,
        d1,
        M2,
        d2,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        M1,
        d1,
        M2,
        d2,
        image_size,
        R,
        T,
        alpha=alpha,
        newImageSize=image_size,
    )

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        M1, d1, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        M2, d2, R2, P2, image_size, cv2.CV_16SC2
    )

    out_dir = rig.calibration_root / rig.rig_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / rig.calibration_npz_name

    np.savez(
        str(out_npz),
        rig_id=rig.rig_id,
        image_width=w,
        image_height=h,
        cameraMatrix1=M1,
        distCoeffs1=d1,
        cameraMatrix2=M2,
        distCoeffs2=d2,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi1=np.array(roi1, dtype=np.int32),
        roi2=np.array(roi2, dtype=np.int32),
        map1_l=map1_l,
        map2_l=map2_l,
        map1_r=map1_r,
        map2_r=map2_r,
        chessboard_inner_cols=cols,
        chessboard_inner_rows=rows,
        square_size_mm=rig.square_size_mm,
        physical_baseline_mm=rig.physical_baseline_mm,
    )

    baseline_mm = float(np.linalg.norm(T) * 1000.0)
    manifest_path = out_dir / "stereo_calib_manifest.json"
    save_rig_manifest(
        manifest_path,
        rig.rig_id,
        (w, h),
        float(rms),
        baseline_mm,
        rig.physical_baseline_mm,
    )

    print(f"Used {used} stereo pairs from {pairs_dir}")
    print(f"Stereo RMS reprojection error: {rms:.4f}")
    print(f"||T|| (baseline from calibration): {baseline_mm:.2f} mm")
    if rig.physical_baseline_mm > 0:
        diff = abs(baseline_mm - rig.physical_baseline_mm)
        print(
            f"Rig physical_baseline_mm: {rig.physical_baseline_mm:.2f} mm "
            f"(|delta| = {diff:.2f} mm — large delta may mean wrong square_size_mm or poor poses)"
        )
    print(f"Saved: {out_npz}")
    print(f"Saved: {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stereo chessboard calibration from left/right image pairs."
    )
    parser.add_argument(
        "--rig",
        type=Path,
        required=True,
        help="Path to rig JSON (see configs/default_rig.example.json).",
    )
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        required=True,
        help="Directory with matching *_{left,right}.jpg|png files.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="stereoRectify alpha (0=crop invalid pixels, 1=keep all).",
    )
    args = parser.parse_args()
    rig = load_rig(args.rig)
    return run_calibration(rig, args.pairs_dir, args.alpha)


if __name__ == "__main__":
    raise SystemExit(main())
