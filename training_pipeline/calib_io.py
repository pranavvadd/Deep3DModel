"""Load stereo_calib.npz (from calibration/calibrate.py) and expose standard keys."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class StereoCalibration:
    """Rectification and intrinsics from OpenCV stereo calibration (saved NPZ)."""

    mtxL: np.ndarray
    distL: np.ndarray
    mtxR: np.ndarray
    distR: np.ndarray
    R: np.ndarray
    T: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    stereo_rms: Optional[float]
    reproj_err_left: Optional[float]
    reproj_err_right: Optional[float]
    baseline_mm: float

    @staticmethod
    def from_npz(path: Path | str) -> "StereoCalibration":
        p = Path(path)
        d = np.load(p, allow_pickle=True)
        f = d.files

        def get_arr(name: str) -> np.ndarray:
            if name not in f:
                raise KeyError(
                    f"stereo calib {p} is missing {name!r} — re-run calibration/calibrate.py"
                )
            return d[name]

        T = get_arr("T")
        # Norm of T in same units as calibration (calibrate.py uses SQUARE_SIZE in mm).
        baseline_mm = float(np.linalg.norm(T))

        def get_float(name: str) -> Optional[float]:
            if name not in f:
                return None
            v = d[name]
            if isinstance(v, np.ndarray) and v.shape == ():
                return float(v)
            if isinstance(v, (float, int)):
                return float(v)
            return float(np.asarray(v).ravel()[0])

        return StereoCalibration(
            mtxL=get_arr("mtxL"),
            distL=get_arr("distL"),
            mtxR=get_arr("mtxR"),
            distR=get_arr("distR"),
            R=get_arr("R"),
            T=T,
            R1=get_arr("R1"),
            R2=get_arr("R2"),
            P1=get_arr("P1"),
            P2=get_arr("P2"),
            Q=get_arr("Q"),
            stereo_rms=get_float("stereo_rms"),
            reproj_err_left=get_float("reproj_err_left"),
            reproj_err_right=get_float("reproj_err_right"),
            baseline_mm=baseline_mm,
        )


def rms_log_lines(cal: StereoCalibration) -> list[str]:
    """Human-readable lines for log output (reprojection + stereo fit from calibration)."""
    lines: list[str] = [
        f"Estimated stereo baseline: {cal.baseline_mm:.3f} mm (||T|| from calib)",
    ]
    if cal.stereo_rms is not None:
        lines.append(f"Calibration stereo RMS error (stereoCalibrate): {cal.stereo_rms:.4f} px")
    if cal.reproj_err_left is not None:
        lines.append(
            f"Per-view mean reprojection error, left:  {cal.reproj_err_left:.4f} px (from calib run)"
        )
    if cal.reproj_err_right is not None:
        lines.append(
            f"Per-view mean reprojection error, right: {cal.reproj_err_right:.4f} px (from calib run)"
        )
    return lines
