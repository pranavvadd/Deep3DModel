# Raspberry Pi USB Camera Recorder (Standalone)

This folder is intentionally standalone and not integrated into the existing Deep3D pipeline.

## What it does
- Records video from a USB UVC camera (including 1080p wide-angle webcams)
- Works on Raspberry Pi 5 and other systems supported by OpenCV
- Saves recordings to a local folder with timestamped filenames

## 1) Install dependency
Use your Python environment (system Python or `venv`) and install OpenCV:

```bash
pip install opencv-python
```

For some Raspberry Pi setups, `opencv-python` wheels may not be available. If that happens, install OpenCV using apt or your preferred Pi method.

## 2) Record video
Run from this folder:

```bash
python recorder.py --camera-index 0 --width 1920 --height 1080 --fps 30 --codec MJPG --preview
```

Press `Ctrl+C` to stop. If preview is enabled, you can also press `q`.

## Common options
- `--camera-index`: camera index (default `0`)
- `--width` / `--height`: requested capture size (default `1920x1080`)
- `--fps`: requested FPS (default `30`)
- `--codec`: output codec (`MJPG`, `XVID`, `mp4v`, etc.)
- `--output-dir`: output folder (default `./recordings`)
- `--output-file`: fixed output filename
- `--max-seconds`: auto-stop duration in seconds (`0` = no limit)
- `--preview`: show live preview window

## Example: fixed filename and 60-second clip
```bash
python recorder.py --output-file test_clip.avi --max-seconds 60
```

## Notes for Raspberry Pi + USB2 camera
- Many USB2 UVC cameras are more stable at 1080p when capture uses MJPG stream; this script requests that automatically.
- Actual camera resolution/FPS are printed at startup because some cameras fall back to lower settings depending on bandwidth and lighting.
