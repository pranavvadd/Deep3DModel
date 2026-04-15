#!/usr/bin/env python3
"""
Touchscreen-friendly recorder controller for Raspberry Pi.

Designed for small displays (such as 3.5" 480x320 panels) with large controls.
Tap "Start Recording" to begin camera capture.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, Button, Frame, Label, StringVar, Tk

from recorder import RecorderConfig, run_recording


class TouchRecorderApp:
    def __init__(self, root: Tk, config: RecorderConfig) -> None:
        self.root = root
        self.config = config
        self._record_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._status = StringVar(value="Ready")

        self.root.title("Pi Touch Camera Recorder")
        self.root.configure(bg="#111111")
        self.root.geometry("480x320")
        self.root.minsize(420, 280)

        container = Frame(root, bg="#111111")
        container.pack(fill=BOTH, expand=True, padx=12, pady=12)

        title = Label(
            container,
            text="Camera Recorder",
            fg="white",
            bg="#111111",
            font=("Helvetica", 22, "bold"),
        )
        title.pack(pady=(8, 20))

        status = Label(
            container,
            textvariable=self._status,
            fg="#dddddd",
            bg="#111111",
            font=("Helvetica", 14),
        )
        status.pack(pady=(0, 20))

        buttons = Frame(container, bg="#111111")
        buttons.pack(fill=BOTH, expand=True)

        self.start_btn = Button(
            buttons,
            text="Start Recording",
            font=("Helvetica", 18, "bold"),
            bg="#2e7d32",
            fg="white",
            activebackground="#388e3c",
            activeforeground="white",
            padx=14,
            pady=24,
            command=self.start_recording,
        )
        self.start_btn.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 8))

        self.stop_btn = Button(
            buttons,
            text="Stop Recording",
            font=("Helvetica", 18, "bold"),
            bg="#b71c1c",
            fg="white",
            activebackground="#c62828",
            activeforeground="white",
            padx=14,
            pady=24,
            command=self.stop_recording,
            state="disabled",
        )
        self.stop_btn.pack(side=RIGHT, fill=BOTH, expand=True, padx=(8, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_recording(self) -> None:
        if self._record_thread and self._record_thread.is_alive():
            return

        self._stop_event.clear()
        self._status.set("Recording...")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self._record_thread = threading.Thread(target=self._record_worker, daemon=True)
        self._record_thread.start()

    def stop_recording(self) -> None:
        if self._record_thread and self._record_thread.is_alive():
            self._status.set("Stopping...")
            self._stop_event.set()

    def _record_worker(self) -> None:
        try:
            run_recording(self.config, stop_event=self._stop_event)
            self.root.after(0, lambda: self._status.set("Saved. Ready"))
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda: self._status.set(f"Error: {exc}"))
        finally:
            self.root.after(0, self._reset_buttons)

    def _reset_buttons(self) -> None:
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def on_close(self) -> None:
        self._stop_event.set()
        self.root.destroy()


def parse_args() -> RecorderConfig:
    parser = argparse.ArgumentParser(description="Touchscreen UI for Pi USB camera recorder.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index.")
    parser.add_argument("--device", type=str, default="", help="Optional camera device path, e.g. /dev/video0.")
    parser.add_argument("--width", type=int, default=1920, help="Requested width.")
    parser.add_argument("--height", type=int, default=1080, help="Requested height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS.")
    parser.add_argument("--codec", type=str, default="MJPG", help="Output writer codec (4 chars).")
    parser.add_argument("--capture-fourcc", type=str, default="MJPG", help="Capture pixel format (4 chars).")
    parser.add_argument("--output-dir", type=Path, default=Path("./recordings"), help="Directory for saved videos.")
    parser.add_argument("--max-seconds", type=int, default=0, help="Auto-stop duration. 0 means no limit.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show OpenCV preview while recording (useful for setup, not required for touchscreen use).",
    )
    args = parser.parse_args()

    return RecorderConfig(
        camera_index=args.camera_index,
        device=args.device.strip() or None,
        width=args.width,
        height=args.height,
        fps=args.fps,
        codec=args.codec,
        capture_fourcc=args.capture_fourcc,
        output_dir=args.output_dir,
        output_file=None,
        preview=args.preview,
        max_seconds=max(args.max_seconds, 0),
        exposure_mode="auto",
        exposure_value=None,
        profile="default",
    )


def main() -> int:
    config = parse_args()
    root = Tk()
    TouchRecorderApp(root, config)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
