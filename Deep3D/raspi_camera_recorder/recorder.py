#!/usr/bin/env python3
"""
Standalone USB UVC camera recorder for Raspberry Pi 5 (and other platforms).

Includes a practical defaults path for the Innomaker U20CAM-1080P (USB2 UVC,
MJPEG/YUY2): use Linux V4L2, request MJPEG for bandwidth, and optional manual
exposure when auto exposure blows out in bright scenes (see vendor FAQ).

Features:
- 1080p capture request (with fallback if camera/driver cannot provide it)
- Optional preview window
- Timestamped output file naming
- FPS/codec/device configurable from CLI
"""

from __future__ import annotations

import argparse
import platform
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2


@dataclass
class RecorderConfig:
    camera_index: int
    device: Optional[str]
    width: int
    height: int
    fps: int
    codec: str
    capture_fourcc: str
    output_dir: Path
    output_file: Optional[str]
    preview: bool
    max_seconds: int
    exposure_mode: str
    exposure_value: Optional[float]
    profile: str


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


def parse_args() -> RecorderConfig:
    parser = argparse.ArgumentParser(
        description="Record from USB UVC camera (e.g. Innomaker U20CAM-1080P)."
    )
    parser.add_argument(
        "--profile",
        choices=("default", "innomaker-u20"),
        default="default",
        help="innomaker-u20: USB2 UVC-friendly defaults (1080p MJPEG, V4L2 tweaks).",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (ignored if --device is set).")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="V4L2 device path, e.g. /dev/video0 (recommended on Pi if index 0 is wrong node).",
    )
    parser.add_argument("--width", type=int, default=1920, help="Requested width.")
    parser.add_argument("--height", type=int, default=1080, help="Requested height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS.")
    parser.add_argument(
        "--codec",
        type=str,
        default="MJPG",
        help="4-char video codec for writer (e.g. MJPG, XVID, mp4v).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./recordings"),
        help="Directory for saved videos.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional fixed output filename (e.g. session.mp4).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live preview window. Press q to stop.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=0,
        help="Auto-stop after N seconds (0 means no limit).",
    )
    parser.add_argument(
        "--capture-fourcc",
        type=str,
        default="MJPG",
        help="Pixel format to request from the driver (4 chars), e.g. MJPG or YUY2.",
    )
    parser.add_argument(
        "--exposure-mode",
        choices=("auto", "manual"),
        default="auto",
        help="manual can reduce outdoor blow-out on some UVC modules (driver-specific).",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="With --exposure-mode manual, V4L2 exposure value (try -1 to -10; driver-specific).",
    )
    args = parser.parse_args()

    codec = args.codec.strip()
    if len(codec) != 4:
        raise ValueError("--codec must be exactly 4 characters, e.g. MJPG or mp4v")

    capture_fourcc = args.capture_fourcc.strip()
    if len(capture_fourcc) != 4:
        raise ValueError("--capture-fourcc must be exactly 4 characters, e.g. MJPG or YUY2")

    device = args.device.strip() or None
    profile = args.profile
    width, height, fps = args.width, args.height, args.fps
    exposure_mode = args.exposure_mode
    exposure_value = args.exposure

    return RecorderConfig(
        camera_index=args.camera_index,
        device=device,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
        capture_fourcc=capture_fourcc,
        output_dir=args.output_dir,
        output_file=args.output_file,
        preview=args.preview,
        max_seconds=max(args.max_seconds, 0),
        exposure_mode=exposure_mode,
        exposure_value=exposure_value,
        profile=profile,
    )


def select_capture_backend() -> int:
    system = platform.system().lower()
    if system == "linux":
        return cv2.CAP_V4L2
    if system == "windows":
        return cv2.CAP_DSHOW
    if system == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def _camera_source(config: RecorderConfig) -> Union[str, int]:
    if config.device:
        return config.device
    return config.camera_index


def _apply_uvc_exposure(cap: cv2.VideoCapture, mode: str, exposure_value: Optional[float]) -> None:
    """Best-effort V4L2 UVC exposure; values are driver-specific."""
    if mode != "manual":
        return
    # Common V4L2 mapping via OpenCV: manual mode then absolute exposure.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    if exposure_value is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)


def open_camera(config: RecorderConfig) -> cv2.VideoCapture:
    backend = select_capture_backend()
    source = _camera_source(config)
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source, cv2.CAP_ANY)
    if not cap.isOpened() and config.device is None:
        cap = cv2.VideoCapture(config.camera_index, cv2.CAP_ANY)

    if not cap.isOpened():
        hint = (
            f"Could not open camera source {source!r}. "
            "On Linux, run `v4l2-ctl --list-devices` and pass the capture node "
            "with --device /dev/videoN (UVC often exposes several nodes)."
        )
        if config.profile == "innomaker-u20":
            hint += " Innomaker U20CAM-1080P is USB UVC (not CSI); use this recorder, not Picamera2."
        raise RuntimeError(hint)

    fourcc = cv2.VideoWriter_fourcc(*config.capture_fourcc[:4])
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS, config.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _apply_uvc_exposure(cap, config.exposure_mode, config.exposure_value)

    return cap


def actual_capture_settings(cap: cv2.VideoCapture) -> Tuple[int, int, float]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    return w, h, fps


def build_output_path(config: RecorderConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.output_file:
        return config.output_dir / config.output_file
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.output_dir / f"uvc_recording_{stamp}.avi"


def create_writer(config: RecorderConfig, frame_size: Tuple[int, int], out_path: Path):
    fourcc = cv2.VideoWriter_fourcc(*config.codec)
    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        float(config.fps),
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open output file {out_path}. Try a different codec "
            "(e.g. mp4v/XVID/MJPG) or file extension."
        )
    return writer


def run_recording(config: RecorderConfig, stop_event: Optional[threading.Event] = None) -> int:
    cap = open_camera(config)
    try:
        actual_w, actual_h, actual_fps = actual_capture_settings(cap)
        out_path = build_output_path(config)
        writer = create_writer(config, (actual_w, actual_h), out_path)

        print("Camera opened successfully.")
        print(
            f"Requested: {config.width}x{config.height}@{config.fps} | "
            f"Actual: {actual_w}x{actual_h}@{actual_fps:.2f}"
        )
        if config.profile == "innomaker-u20" and not config.device:
            print(
                "Innomaker U20 tip: if the image is wrong or black, run "
                "`v4l2-ctl --list-devices` and use --device /dev/videoN for the "
                "UVC capture node (not the metadata node)."
            )
        if config.profile == "innomaker-u20":
            print(
                "Outdoor overexposure: try --exposure-mode manual --exposure -4 "
                "(adjust -2 to -10; driver-specific)."
            )
        print(f"Recording to: {out_path}")
        print("Press Ctrl+C to stop.")
        if config.preview:
            print("Preview enabled: press q in preview window to stop.")

        stopper = GracefulStop()
        start_time = time.time()
        frames = 0

        while not stopper.should_stop:
            if stop_event is not None and stop_event.is_set():
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed; stopping.")
                break

            writer.write(frame)
            frames += 1

            if config.preview:
                cv2.imshow("UVC Camera Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if config.max_seconds > 0 and (time.time() - start_time) >= config.max_seconds:
                break

        elapsed = max(time.time() - start_time, 1e-6)
        print(f"Finished. Captured {frames} frames in {elapsed:.2f}s ({frames / elapsed:.2f} FPS).")
        print(f"Saved video: {out_path}")
        return 0
    finally:
        cap.release()
        try:
            writer.release()  # type: ignore[name-defined]
        except NameError:
            pass
        cv2.destroyAllWindows()


def main() -> int:
    try:
        config = parse_args()
        return run_recording(config)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
