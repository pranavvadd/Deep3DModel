#!/usr/bin/env python3
"""
Stereo capture using Picamera2 / libcamera on Raspberry Pi (dual CSI cameras).

Two independent Picamera2 instances (camera_num left/right). Frames are captured
back-to-back each iteration; hardware stereo sync is not assumed—use wall-clock
timestamp logs and scheduling (RTC-backed system time) rather than frame-level
time alignment tricks.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from rig_config import RigConfig, load_rectify_maps, load_rig
from wall_clock import (
    pair_timestamp,
    parse_start_at,
    sleep_until,
    write_session_json,
)

try:
    from picamera2 import Picamera2
except ImportError as exc:  # pragma: no cover - Pi-only dependency
    print(
        "Error: picamera2 is not installed or this is not a Raspberry Pi OS "
        "environment with libcamera.\n"
        "On Raspberry Pi: sudo apt install -y python3-picamera2\n"
        "Or: pip install picamera2 (may require matching system packages).\n"
        "USB UVC modules (e.g. Innomaker U20CAM-1080P) are not CSI cameras: use "
        "raspi_camera_recorder/recorder.py with --profile innomaker-u20 instead.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


@dataclass
class StereoConfig:
    camera_left: int
    camera_right: int
    width: int
    height: int
    fps: float
    mode: str  # "video" | "stills"
    still_count: int
    output_dir: Path
    prefix: str
    codec: str
    preview: bool
    max_seconds: int
    warmup_seconds: float
    rig_id: str
    rectify: bool
    calibration_npz: Optional[Path]
    timestamps_csv: Optional[Path]
    start_at: Optional[datetime]
    pace_fps: bool
    session_json: bool


class GracefulStop:
    def __init__(self) -> None:
        self._stop = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame) -> None:  # noqa: ARG002
        self._stop = True

    @property
    def should_stop(self) -> bool:
        return self._stop


def parse_args() -> StereoConfig:
    parser = argparse.ArgumentParser(
        description="Stereo capture with Picamera2 (libcamera), dual CSI cameras."
    )
    parser.add_argument(
        "--rig",
        type=Path,
        default=None,
        help="Rig JSON profile (indices, resolution, calibration paths). CLI overrides file.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Stereo calibration NPZ (defaults to rig calibration_output if --rig set).",
    )
    parser.add_argument(
        "--rectify",
        action="store_true",
        help="Rectify frames using calibration NPZ (must match capture resolution).",
    )
    parser.add_argument(
        "--camera-left",
        type=int,
        default=None,
        help="libcamera index for left camera (override rig).",
    )
    parser.add_argument(
        "--camera-right",
        type=int,
        default=None,
        help="libcamera index for right camera (override rig).",
    )
    parser.add_argument("--width", type=int, default=None, help="Override rig / default 1920.")
    parser.add_argument("--height", type=int, default=None, help="Override rig / default 1080.")
    parser.add_argument("--fps", type=float, default=None, help="Override rig / default 30.")
    parser.add_argument(
        "--mode",
        choices=("video", "stills"),
        default="video",
        help="video: write left/right AVI; stills: JPEG pairs.",
    )
    parser.add_argument(
        "--still-count",
        type=int,
        default=10,
        help="Number of stereo pairs in stills mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./stereo_recordings"),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional filename prefix (default: timestamp).",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="MJPG",
        help="OpenCV VideoWriter fourcc for video mode (e.g. MJPG, XVID, mp4v).",
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=0,
        help="Stop after N seconds in video mode (0 = until Ctrl+C).",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=2.0,
        help="Sleep after start() before first capture (AE/AWB settle).",
    )
    parser.add_argument(
        "--timestamps-csv",
        type=Path,
        default=None,
        help="Append one row per stereo pair: wall UTC (ISO), unix_ns, monotonic_ns.",
    )
    parser.add_argument(
        "--start-at",
        type=str,
        default=None,
        help="Wait until this wall time before opening cameras (ISO8601; use Z for UTC).",
    )
    parser.add_argument(
        "--no-pace-fps",
        action="store_true",
        help="In video mode, do not sleep to hit --fps (max rate; use with timestamps CSV).",
    )
    parser.add_argument(
        "--session-json",
        action="store_true",
        help="Write {prefix}_session.json summary (start/end wall time, pair count).",
    )
    args = parser.parse_args()

    codec = args.codec.strip()
    if len(codec) != 4:
        raise ValueError("--codec must be exactly 4 characters")

    prefix = args.prefix.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

    rig: Optional[RigConfig] = load_rig(args.rig) if args.rig else None

    camera_left = (
        args.camera_left if args.camera_left is not None else (rig.left_camera_index if rig else 0)
    )
    camera_right = (
        args.camera_right if args.camera_right is not None else (rig.right_camera_index if rig else 1)
    )
    width = args.width if args.width is not None else (rig.width if rig else 1920)
    height = args.height if args.height is not None else (rig.height if rig else 1080)
    fps = args.fps if args.fps is not None else (rig.fps if rig else 30.0)

    calib_npz: Optional[Path] = None
    if args.rectify:
        calib_npz = args.calibration
        if calib_npz is None and rig is not None:
            calib_npz = rig.calibration_npz_path()
        if calib_npz is not None:
            calib_npz = calib_npz.expanduser().resolve()
        if calib_npz is None or not calib_npz.is_file():
            raise ValueError(
                "--rectify needs an existing calibration NPZ. "
                "Pass --calibration path/to/stereo.npz or use --rig whose calibration_output points to one."
            )

    rig_id = rig.rig_id if rig else ""

    start_at: Optional[datetime] = None
    if args.start_at:
        start_at = parse_start_at(args.start_at)

    timestamps_csv: Optional[Path] = None
    if args.timestamps_csv is not None:
        timestamps_csv = args.timestamps_csv.expanduser().resolve()

    return StereoConfig(
        camera_left=camera_left,
        camera_right=camera_right,
        width=width,
        height=height,
        fps=max(fps, 1.0),
        mode=args.mode,
        still_count=max(args.still_count, 1),
        output_dir=args.output_dir,
        prefix=prefix,
        codec=codec,
        preview=args.preview,
        max_seconds=max(args.max_seconds, 0),
        warmup_seconds=max(args.warmup_seconds, 0.0),
        rig_id=rig_id,
        rectify=args.rectify,
        calibration_npz=calib_npz,
        timestamps_csv=timestamps_csv,
        start_at=start_at,
        pace_fps=not args.no_pace_fps,
        session_json=args.session_json,
    )


def _build_video_config(picam: Picamera2, width: int, height: int) -> dict:
    return picam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        buffer_count=4,
    )


def _array_to_bgr(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        return arr
    c = arr.shape[2]
    if c == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if c == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def _rectify_pair(
    bgr_l: np.ndarray,
    bgr_r: np.ndarray,
    maps: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    ew, eh, m1l, m2l, m1r, m2r = maps
    w, h = bgr_l.shape[1], bgr_l.shape[0]
    if (w, h) != (ew, eh):
        raise RuntimeError(
            f"Calibration NPZ is for {ew}x{eh} but frames are {w}x{h}. "
            "Recalibrate at this resolution or match rig capture size."
        )
    out_l = cv2.remap(bgr_l, m1l, m2l, cv2.INTER_LINEAR)
    out_r = cv2.remap(bgr_r, m1r, m2r, cv2.INTER_LINEAR)
    return out_l, out_r


def _schedule_wall_start(config: StereoConfig) -> None:
    if config.start_at is None:
        return
    now = datetime.now(timezone.utc)
    print(
        f"Waiting for wall time {config.start_at.isoformat()} "
        f"(now {now.isoformat()})."
    )
    sleep_until(config.start_at)
    print("Wall start time reached; opening cameras.")


def _init_timestamps_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "frame_index,pair_utc_iso,unix_ns,monotonic_ns\n",
        encoding="utf-8",
    )


def _append_pair_csv(path: Path, frame_index: int, iso: str, unix_ns: int, mono_ns: int) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{frame_index},{iso},{unix_ns},{mono_ns}\n")


def _try_set_frame_rate(left: Picamera2, right: Picamera2, fps: float) -> None:
    try:
        period = int(1_000_000 / fps)
        controls = {"FrameDurationLimits": (period, period)}
        left.set_controls(controls)
        right.set_controls(controls)
    except Exception:
        pass


def run_stills(config: StereoConfig) -> int:
    _schedule_wall_start(config)
    started_wall = datetime.now(timezone.utc).isoformat()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    maps: Optional[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    if config.rectify and config.calibration_npz:
        maps = load_rectify_maps(config.calibration_npz)
        print(f"Rectification enabled: {config.calibration_npz}")
    if config.rig_id:
        print(f"Rig: {config.rig_id}")

    if config.timestamps_csv is not None:
        _init_timestamps_csv(config.timestamps_csv)
        print(f"Pair timestamps -> {config.timestamps_csv}")

    left = Picamera2(config.camera_left)
    right = Picamera2(config.camera_right)

    cfg_l = _build_video_config(left, config.width, config.height)
    cfg_r = _build_video_config(right, config.width, config.height)
    left.configure(cfg_l)
    right.configure(cfg_r)

    left.start()
    right.start()
    try:
        if config.warmup_seconds:
            time.sleep(config.warmup_seconds)
        _try_set_frame_rate(left, right, config.fps)

        for i in range(config.still_count):
            path_l = config.output_dir / f"{config.prefix}_{i:04d}_left.jpg"
            path_r = config.output_dir / f"{config.prefix}_{i:04d}_right.jpg"
            if maps is None:
                left.capture_file(str(path_l))
                right.capture_file(str(path_r))
            else:
                arr_l = _array_to_bgr(left.capture_array("main"))
                arr_r = _array_to_bgr(right.capture_array("main"))
                arr_l, arr_r = _rectify_pair(arr_l, arr_r, maps)
                cv2.imwrite(str(path_l), arr_l)
                cv2.imwrite(str(path_r), arr_r)
            if config.timestamps_csv is not None:
                iso, u_ns, m_ns = pair_timestamp()
                _append_pair_csv(config.timestamps_csv, i, iso, u_ns, m_ns)
            print(f"Saved pair {i + 1}/{config.still_count}: {path_l.name}, {path_r.name}")

        ended_wall = datetime.now(timezone.utc).isoformat()
        if config.session_json:
            write_session_json(
                config.output_dir / f"{config.prefix}_session.json",
                {
                    "rig_id": config.rig_id,
                    "prefix": config.prefix,
                    "mode": "stills",
                    "time_basis": "OS CLOCK_REALTIME (Pi RTC + NTP as configured; one RTC module)",
                    "started_utc_iso": started_wall,
                    "ended_utc_iso": ended_wall,
                    "stereo_pairs": config.still_count,
                    "timestamps_csv": str(config.timestamps_csv)
                    if config.timestamps_csv
                    else None,
                },
            )
        print(f"Done. Output directory: {config.output_dir}")
        return 0
    finally:
        left.stop()
        right.stop()
        left.close()
        right.close()


def run_video(config: StereoConfig) -> int:
    _schedule_wall_start(config)
    started_wall = datetime.now(timezone.utc).isoformat()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    maps: Optional[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    if config.rectify and config.calibration_npz:
        maps = load_rectify_maps(config.calibration_npz)
        print(f"Rectification enabled: {config.calibration_npz}")
    if config.rig_id:
        print(f"Rig: {config.rig_id}")

    if config.timestamps_csv is not None:
        _init_timestamps_csv(config.timestamps_csv)
        print(f"Pair timestamps -> {config.timestamps_csv}")

    left = Picamera2(config.camera_left)
    right = Picamera2(config.camera_right)

    cfg_l = _build_video_config(left, config.width, config.height)
    cfg_r = _build_video_config(right, config.width, config.height)
    left.configure(cfg_l)
    right.configure(cfg_r)

    fourcc = cv2.VideoWriter_fourcc(*config.codec)
    out_l = config.output_dir / f"{config.prefix}_left.avi"
    out_r = config.output_dir / f"{config.prefix}_right.avi"
    writer_l: Optional[cv2.VideoWriter] = None
    writer_r: Optional[cv2.VideoWriter] = None

    left.start()
    right.start()
    stopper = GracefulStop()
    start_time = time.time()
    frames = 0

    try:
        if config.warmup_seconds:
            time.sleep(config.warmup_seconds)
        _try_set_frame_rate(left, right, config.fps)

        print("Stereo video recording. Ctrl+C to stop.")
        if config.preview:
            print("Preview: press q to stop.")
        if not config.pace_fps:
            print("FPS pacing disabled (--no-pace-fps); capture runs as fast as the pipeline allows.")

        frame_interval = 1.0 / config.fps if config.pace_fps else 0.0

        while not stopper.should_stop:
            loop_start = time.time()

            arr_l = left.capture_array("main")
            arr_r = right.capture_array("main")

            bgr_l = _array_to_bgr(arr_l)
            bgr_r = _array_to_bgr(arr_r)
            if maps is not None:
                bgr_l, bgr_r = _rectify_pair(bgr_l, bgr_r, maps)
            h, w = bgr_l.shape[:2]
            if writer_l is None:
                writer_l = cv2.VideoWriter(
                    str(out_l), fourcc, config.fps, (w, h)
                )
                writer_r = cv2.VideoWriter(
                    str(out_r), fourcc, config.fps, (w, h)
                )
                if not writer_l.isOpened() or not writer_r.isOpened():
                    raise RuntimeError(
                        f"Could not open video writers. Try --codec XVID or mp4v. "
                        f"Paths: {out_l}, {out_r}"
                    )
                print(f"Writing: {out_l.name}, {out_r.name}")
                meta_l = left.capture_metadata()
                print(
                    f"Resolution {w}x{h} @ target {config.fps} FPS "
                    f"(left metadata sample: {meta_l})"
                )

            if config.timestamps_csv is not None:
                iso, u_ns, m_ns = pair_timestamp()
                _append_pair_csv(config.timestamps_csv, frames, iso, u_ns, m_ns)

            writer_l.write(bgr_l)
            writer_r.write(bgr_r)
            frames += 1

            if config.preview:
                combined = np.hstack([bgr_l, bgr_r])
                cv2.imshow("Stereo (L | R)", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if config.max_seconds > 0 and (time.time() - start_time) >= config.max_seconds:
                break

            if config.pace_fps and frame_interval > 0:
                elapsed = time.time() - loop_start
                slack = frame_interval - elapsed
                if slack > 0:
                    time.sleep(slack)

        elapsed = max(time.time() - start_time, 1e-6)
        print(f"Captured {frames} frame pairs in {elapsed:.2f}s ({frames / elapsed:.2f} pairs/s).")
        print(f"Saved: {out_l}\n       {out_r}")
        ended_wall = datetime.now(timezone.utc).isoformat()
        if config.session_json:
            write_session_json(
                config.output_dir / f"{config.prefix}_session.json",
                {
                    "rig_id": config.rig_id,
                    "prefix": config.prefix,
                    "mode": "video",
                    "time_basis": "OS CLOCK_REALTIME (Pi RTC + NTP as configured; one RTC module)",
                    "started_utc_iso": started_wall,
                    "ended_utc_iso": ended_wall,
                    "stereo_pairs": frames,
                    "pace_fps": config.pace_fps,
                    "target_fps": config.fps,
                    "timestamps_csv": str(config.timestamps_csv)
                    if config.timestamps_csv
                    else None,
                },
            )
        return 0
    finally:
        if writer_l is not None:
            writer_l.release()
        if writer_r is not None:
            writer_r.release()
        left.stop()
        right.stop()
        left.close()
        right.close()
        cv2.destroyAllWindows()


def main() -> int:
    try:
        config = parse_args()
        if config.mode == "stills":
            return run_stills(config)
        return run_video(config)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
