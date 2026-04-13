#!/usr/bin/env python3
"""
Wiring reference for Raspberry Pi 40-pin header (Pi 4 / Pi 5 layout).

This module does not configure GPIO or I2C at runtime. Picamera2 uses CSI
connectors, not these pins. Values here document how to wire an optional
I2C RTC (or other 3.3 V I2C peripheral) if you add one.

Enable I2C: sudo raspi-config → Interface Options → I2C → reboot.
"""

from __future__ import annotations

# --- Optional I2C RTC (typical DS3231-style breakout), I2C bus 1 ---

# Supply: use 3.3 V unless the module datasheet explicitly requires 5 V.
I2C_PERIPHERAL_SUPPLY_V = 3.3

# Physical pin numbers on the 40-pin header (count from pin 1 at board corner).
PHYSICAL_PIN_3V3 = 1
PHYSICAL_PIN_I2C1_SDA = 3
PHYSICAL_PIN_I2C1_SCL = 5
PHYSICAL_PIN_GND = 6  # any GND pin is valid; 6 is next to SCL

# Broadcom GPIO numbers (what Linux / overlays use for labeling).
BCM_GPIO_I2C1_SDA = 2
BCM_GPIO_I2C1_SCL = 3

# Default I2C device path on Raspberry Pi OS (bus 1 on many models).
I2C_BUS_DEVICE_DEFAULT = "/dev/i2c-1"

# --- Raspberry Pi Camera Module 3 (stereo) ---

# No GPIO header pins: each module uses a dedicated CSI flex connector.
# libcamera indices (0, 1, …) are set in the rig JSON, not pin numbers.
CAMERA_INTERFACE = "CSI (ribbon cable to CAM/DISP port, not 40-pin header)"


def wiring_summary() -> str:
    return (
        "Optional I2C RTC (3.3 V I2C1):\n"
        f"  3.3V  → physical pin {PHYSICAL_PIN_3V3}\n"
        f"  GND   → physical pin {PHYSICAL_PIN_GND} (or any GND)\n"
        f"  SDA   → physical pin {PHYSICAL_PIN_I2C1_SDA} (BCM GPIO{BCM_GPIO_I2C1_SDA})\n"
        f"  SCL   → physical pin {PHYSICAL_PIN_I2C1_SCL} (BCM GPIO{BCM_GPIO_I2C1_SCL})\n"
        f"  Supply voltage: {I2C_PERIPHERAL_SUPPLY_V} V (use module datasheet if different).\n"
        "\n"
        f"Camera Module 3: {CAMERA_INTERFACE}\n"
        "  Use rig JSON camera_index + rpicam-hello --list-cameras.\n"
    )


if __name__ == "__main__":
    print(wiring_summary())
