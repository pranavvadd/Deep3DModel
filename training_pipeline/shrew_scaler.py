"""
Scale disparity / depth maps for a different (subject) inter-pupillary distance (IPD).

In rectified pinhole stereo, disparity d (pixels) and depth Z satisfy d ≈ f·B / Z
for camera baseline B and focal f. For a fixed scene (same Z), d is *linear* in B.
A simple anatomical rescaling: treat a reference IPD (e.g. human ~63 mm) as a
"nominal" baseline in image space and rescale to a target species IPD (e.g. tree
shrew ~12 mm) so the model sees parallax amplitudes consistent with that subject.

This is a *first-order* model; it does not replace a full off-axis eye model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DEEP3D = _REPO / "Deep3D"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_DEEP3D) not in sys.path:
    sys.path.insert(0, str(_DEEP3D))

from utils import util  # noqa: E402

Array = np.ndarray


def ipd_disparity_scale(reference_ipd_mm: float, target_ipd_mm: float) -> float:
    """
    Multiplicative scale to apply to disparity when changing “effective” IPD
    (same scene geometry, different baseline/viewer separation).

    d_new = d * (target_ipd / reference_ipd)
    Smaller target IPD → smaller disparity (less parallax) for the same depth.
    """
    if reference_ipd_mm <= 0:
        raise ValueError("reference_ipd_mm must be positive")
    if target_ipd_mm <= 0:
        raise ValueError("target_ipd_mm must be positive")
    return target_ipd_mm / reference_ipd_mm


def scale_disparity(
    disparity: Array,
    scale: float,
    mask_invalid: Optional[Array] = None,
) -> np.ndarray:
    """
    Apply linear scale to a disparity field (float32/64 recommended).

    If *mask_invalid* is provided (bool, same shape), those pixels are set to
    0.0 in the output (SGBM invalid regions are often < 0).
    """
    out = disparity.astype(np.float64) * float(scale)
    if mask_invalid is not None:
        out = out.copy()
        out[mask_invalid] = 0.0
    return out.astype(np.float32)


def scale_for_camera_baseline(
    measured_baseline_mm: float,
    target_ipd_mm: float,
    reference_ipd_mm: float = 63.0,
) -> float:
    """
    First-order scale when disparity was measured with a physical camera baseline
    *measured_baseline_mm* (e.g. ``||T||`` from *stereo_calib.npz*) and you want
    a field that behaves like a viewer (or emulated eye separation) of
    *target_ipd_mm*.

    Under *d* ∝ *B* for fixed scene content, the combined factor simplifies to

        scale = (reference_ipd / B_cam) * (IPD_target / IPD_ref)
              = IPD_target / B_cam  .

    *reference_ipd_mm* is kept in the signature for API clarity; it cancels
    mathematically. Returns the scalar to multiply each disparity (pixels).
    """
    if measured_baseline_mm <= 0:
        raise ValueError("measured_baseline_mm must be positive")
    b_scale = reference_ipd_mm / measured_baseline_mm
    ipd_scale = target_ipd_mm / reference_ipd_mm
    return b_scale * ipd_scale


def process_npy_file(
    in_path: Path, out_path: Path, scale: float, use_invalid_zero: bool = True
) -> None:
    d = np.load(in_path, allow_pickle=True)
    if d.ndim != 2:
        raise ValueError(f"Expected HxW disparity, got {d.shape} for {in_path}")
    if use_invalid_zero:
        invalid = d < 0
    else:
        invalid = None
    out = scale_disparity(d, scale, mask_invalid=invalid)
    util.makedirs(str(out_path.parent))
    np.save(str(out_path), out)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Scale float disparity .npy for IPD (e.g. shrew).")
    p.add_argument("input", type=Path, help="Input .npy (float disparity, pixels).")
    p.add_argument("output", type=Path, help="Output .npy path.")
    p.add_argument(
        "--ref-ipd",
        type=float,
        default=63.0,
        help="Reference IPD in mm (default human).",
    )
    p.add_argument(
        "--target-ipd",
        type=float,
        default=12.0,
        help="Target (e.g. tree shrew) IPD in mm.",
    )
    p.add_argument(
        "--baseline-mm",
        type=float,
        default=None,
        help="Optional: physical camera baseline from calib; uses combined scale.",
    )
    args = p.parse_args()
    _setup_logging()

    if args.baseline_mm is not None:
        s = scale_for_camera_baseline(
            float(args.baseline_mm), float(args.target_ipd), float(args.ref_ipd)
        )
        logging.info(
            "Using baseline-aware scale: baseline=%.3f mm → scale=%.6f", args.baseline_mm, s
        )
    else:
        s = ipd_disparity_scale(args.ref_ipd, args.target_ipd)
        logging.info(
            "IPD-only scale: ref=%.1f mm target=%.1f mm → scale=%.6f", args.ref_ipd, args.target_ipd, s
        )
    process_npy_file(args.input.resolve(), args.output.resolve(), s)
    logging.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
