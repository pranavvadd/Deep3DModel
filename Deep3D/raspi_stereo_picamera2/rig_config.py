#!/usr/bin/env python3
"""Load stereo rig profiles from JSON (swap cameras / baseline without code changes)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class RigConfig:
    rig_id: str
    notes: str
    physical_baseline_mm: float
    left_camera_index: int
    right_camera_index: int
    left_sensor_label: str
    right_sensor_label: str
    width: int
    height: int
    fps: float
    chessboard_inner_cols: int
    chessboard_inner_rows: int
    square_size_mm: float
    calibration_root: Path
    calibration_npz_name: str

    def calibration_npz_path(self) -> Path:
        return self.calibration_root / self.rig_id / self.calibration_npz_name


def _resolve_path(base_dir: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_rig(path: Path) -> RigConfig:
    path = path.expanduser().resolve()
    raw: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "rig_id" not in raw:
        raise ValueError("rig JSON missing required key: rig_id")
    left = raw.get("left") or {}
    right = raw.get("right") or {}
    capture = raw.get("capture") or {}
    cal = raw.get("calibration") or {}
    out = raw.get("calibration_output") or {}

    cal_dir = out.get("directory", "calibrations")
    cal_root = _resolve_path(path.parent, cal_dir)
    npz_name = out.get("filename", "stereo.npz")

    return RigConfig(
        rig_id=str(raw["rig_id"]),
        notes=str(raw.get("notes", "")),
        physical_baseline_mm=float(raw.get("physical_baseline_mm", 0.0)),
        left_camera_index=int(left.get("camera_index", 0)),
        right_camera_index=int(right.get("camera_index", 1)),
        left_sensor_label=str(left.get("sensor_label", "")),
        right_sensor_label=str(right.get("sensor_label", "")),
        width=int(capture.get("width", 1920)),
        height=int(capture.get("height", 1080)),
        fps=float(capture.get("fps", 30.0)),
        chessboard_inner_cols=int(cal.get("chessboard_inner_cols", 9)),
        chessboard_inner_rows=int(cal.get("chessboard_inner_rows", 6)),
        square_size_mm=float(cal.get("square_size_mm", 25.0)),
        calibration_root=cal_root,
        calibration_npz_name=str(npz_name),
    )


def save_rig_manifest(
    path: Path,
    rig_id: str,
    image_size: tuple[int, int],
    rms: float,
    baseline_mm_from_t: float,
    physical_baseline_mm: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rig_id": rig_id,
        "image_size": list(image_size),
        "rms_reprojection_error": rms,
        "baseline_mm_from_translation": baseline_mm_from_t,
        "physical_baseline_mm_from_rig": physical_baseline_mm,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_rectify_maps(
    npz_path: Path,
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(str(npz_path.expanduser().resolve()), allow_pickle=False)
    w = int(data["image_width"])
    h = int(data["image_height"])
    return (
        w,
        h,
        data["map1_l"],
        data["map2_l"],
        data["map1_r"],
        data["map2_r"],
    )
