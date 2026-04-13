#!/usr/bin/env bash
# Raspberry Pi OS: install the supported camera stack for this repo (Picamera2 + libcamera via apt).
# Run on the Pi: bash raspi_stereo_picamera2/scripts/install_pi_os_camera_stack.sh
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this script on Raspberry Pi OS (Linux), not on macOS/Windows."
  exit 1
fi

echo "Updating package lists..."
sudo apt-get update

echo "Installing Picamera2 and libcamera Python bindings from apt (recommended by Raspberry Pi)..."
sudo apt-get install -y \
  python3-picamera2 \
  python3-opencv \
  python3-numpy \
  rpicam-apps

echo "Verifying Picamera2 import..."
python3 -c "from picamera2 import Picamera2; print('Picamera2 import OK')"

echo "Listing cameras (libcamera)..."
rpicam-hello --list-cameras || true

echo "Done. Use python3 (system) to run stereo_capture.py so it picks up apt packages."
