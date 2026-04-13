#!/usr/bin/env python3
"""
Wall-clock helpers for capture scheduling and timestamp logs.

Uses the OS real-time clock (CLOCK_REALTIME): on Raspberry Pi OS this is what
stays correct across reboots when an onboard RTC + battery and/or NTP are
configured. No extra Python RTC driver is required for one onboard module.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def pair_timestamp() -> tuple[str, int, int]:
    """Return (UTC ISO with µs Z, unix_ns, monotonic_ns) for one wall-clock instant."""
    unix_ns = time.time_ns()
    mono_ns = time.monotonic_ns()
    dt = datetime.fromtimestamp(unix_ns / 1e9, tz=timezone.utc)
    iso = dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{dt.microsecond:06d}Z"
    return iso, unix_ns, mono_ns


def parse_start_at(s: str) -> datetime:
    """
    Parse scheduling instant. Naive strings use the machine local timezone.
    Prefer explicit UTC: ...Z or ...+00:00.
    """
    raw = s.strip()
    if raw.endswith("Z"):
        raw = raw[:-1]
        dt = datetime.fromisoformat(raw)
        return dt.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        local = datetime.now().astimezone().tzinfo
        if local is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.replace(tzinfo=local)
    return dt.astimezone(timezone.utc)


def sleep_until(when_utc: datetime) -> None:
    """Block until OS wall time is >= when_utc (poll; fine for scheduling)."""
    target = when_utc.timestamp()
    while True:
        now = time.time()
        if now >= target:
            return
        remaining = target - now
        time.sleep(min(0.25, max(0.001, remaining)))


def write_session_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
