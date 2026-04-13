# Stereo Picamera2 rig (config + calibration)

Standalone module: **dual CSI cameras** on Raspberry Pi via **Picamera2 / libcamera**. USB webcams belong in `../raspi_camera_recorder/`.

The example rig assumes **two [Raspberry Pi Camera Module 3](https://www.raspberrypi.com/products/camera-module-3/)** boards (Sony **IMX708**), which work with the stock Pi OS **libcamera** stack and Picamera2. On a **Pi 5**, use the two CSI ports and 15-pin FFCs (adapters exist for 22-pin legacy cables). Confirm order with:

```bash
rpicam-hello --list-cameras
```

Standard **1920×1080 @ ~30 fps** is a practical default; other modes are available via the sensor mode table—keep **calibration and live capture at the same resolution**.

If you mix **wide** vs **standard** CM3 or add a **noIR** unit, only metadata and calibration need updating; change `sensor_label` in the rig JSON and **re-run stereo calibration**.

## Design goals

- **Comfortable baseline**: set `physical_baseline_mm` in your rig JSON (documentation + sanity check vs. calibrated `||T||`). A common tabletop range is **~100–140 mm** between lens axes; widen for larger scenes.
- **Per-rig calibration**: chessboard solve writes `calibrations/<rig_id>/stereo.npz` + manifest JSON. No code edits when you change cameras—**copy/edit the rig file, re-shoot pairs, re-run calibration**.
- **`sensor_label`**: free-text metadata only; it does not affect math.

## Files

| File | Purpose |
|------|---------|
| `configs/default_rig.example.json` | Copy to e.g. `my_rig.json` and customize |
| `rig_config.py` | Load rig JSON; load rectification maps from NPZ |
| `calibrate_stereo.py` | Build `stereo.npz` from stereo image pairs |
| `stereo_capture.py` | Record / stills; optional `--rig` + `--rectify` |
| `wall_clock.py` | Wall time parsing, wait-until, session JSON helpers |
| `hardware_pins.py` | **Reference constants** for optional I2C RTC wiring (not used by capture at runtime) |
| `scripts/install_pi_os_camera_stack.sh` | **Pi OS**: `apt` install Picamera2 + libcamera-aligned deps for recording |

## Hardware pin reference (in code)

`hardware_pins.py` lists **physical header pin numbers**, **BCM GPIO** for default **I2C1** (SDA/SCL), **3.3 V**, and notes that **Camera Module 3 uses CSI only**. That file does **not** drive pins—the kernel owns I2C once enabled in `raspi-config`. Print the summary on the Pi:

```bash
python3 raspi_stereo_picamera2/hardware_pins.py
```

## Raspberry Pi OS setup for recording (Picamera2 / libcamera)

Use **current Raspberry Pi OS** (e.g. **64-bit Bookworm**) on the Pi. CSI cameras are supported through the **libcamera** pipeline; **Picamera2** is the supported Python layer. Raspberry Pi recommends installing **Picamera2 with `apt`** so Python bindings stay aligned with the system **libcamera** version.

**Do this on the Pi:**

1. **Fully update the OS** (kernel/firmware and camera stack move together):

   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo reboot
   ```

2. **Install the camera stack and OpenCV/NumPy from apt** (not `pip install picamera2` for normal use):

   ```bash
   cd ~/Deep3DModel/Deep3D   # or wherever you cloned the project
   bash raspi_stereo_picamera2/scripts/install_pi_os_camera_stack.sh
   ```

   Equivalent manual one-liner:

   ```bash
   sudo apt install -y python3-picamera2 python3-opencv python3-numpy rpicam-apps
   ```

3. **Check both cameras are visible to libcamera:**

   ```bash
   rpicam-hello --list-cameras
   ```

4. **Confirm Python sees Picamera2 (system interpreter):**

   ```bash
   python3 -c "from picamera2 import Picamera2; print('OK')"
   ```

Run **`python3 raspi_stereo_picamera2/stereo_capture.py ...`** on the Pi so you use the **apt** packages. A **venv** that shadows `picamera2` with an incompatible pip wheel is a common source of failures—avoid mixing unless you know the versions match.

**Legacy note:** Raspberry Pi **Camera Module 3** needs the **libcamera** stack, not the old closed-source camera stack.

## Quick workflow

### 1) Install (on Pi)

If you already followed **Raspberry Pi OS setup for recording** above, skip this. Otherwise:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-numpy rpicam-apps
```

### 2) Define the rig (no code changes later)

```bash
cp raspi_stereo_picamera2/configs/default_rig.example.json raspi_stereo_picamera2/configs/my_rig.json
# Edit: rig_id, camera_index left/right, physical_baseline_mm, sensor_label, capture size, chessboard fields
```

Paths in `calibration_output.directory` are **relative to the rig JSON file’s folder** unless you use an absolute path.

### 3) Capture calibration pairs

Use **stills** mode so filenames match the calibrator (`stem_left.jpg` / `stem_right.jpg`):

```bash
cd Deep3D
python3 raspi_stereo_picamera2/stereo_capture.py \
  --rig raspi_stereo_picamera2/configs/my_rig.json \
  --mode stills \
  --still-count 30 \
  --output-dir raspi_stereo_picamera2/calib_pairs \
  --prefix calib
```

This produces `calib_0000_left.jpg`, `calib_0000_right.jpg`, … Move the board; keep both cameras seeing all inner corners.

### 4) Run calibration

```bash
python3 raspi_stereo_picamera2/calibrate_stereo.py \
  --rig raspi_stereo_picamera2/configs/my_rig.json \
  --pairs-dir raspi_stereo_picamera2/calib_pairs \
  --alpha 0
```

Outputs:

- `calibrations/<rig_id>/stereo.npz` (intrinsics, extrinsics, rectification **maps**)
- `calibrations/<rig_id>/stereo_calib_manifest.json` (RMS, baseline from `T`, your `physical_baseline_mm`)

Compare **baseline_mm_from_translation** to **physical_baseline_mm**; a large mismatch often means wrong `square_size_mm` or bad corner detections.

### 5) Record with rectification

Match **capture resolution** to calibration (same as in the rig / pair images):

```bash
python3 raspi_stereo_picamera2/stereo_capture.py \
  --rig raspi_stereo_picamera2/configs/my_rig.json \
  --rectify \
  --preview \
  --max-seconds 15
```

## Swapping camera models

1. Update `sensor_label` (optional, for your records).
2. Confirm `camera_index` if libcamera order changed (`rpicam-hello --list-cameras`).
3. Re-capture calibration pairs at the **same** `capture.width` / `height` you will use live.
4. Re-run `calibrate_stereo.py`.
5. Keep the same `--rig` path (or point `--calibration` at the new NPZ).

## Stereo pair naming for `calibrate_stereo.py`

In `--pairs-dir`, each pair must share a stem, e.g. `foo_left.jpg` and `foo_right.jpg`. Stills mode uses `{prefix}_{index:04d}_left.jpg` / `_right.jpg`.

## Wall clock, RTC, and scheduling (one clock for the whole rig)

Timestamping and **start scheduling** use the OS **CLOCK_REALTIME** (`time.time_ns()` / UTC in logs). You do **not** need a second RTC in software: a **single** battery-backed RTC (often already on the Pi or HAT) keeps **system time** valid across power loss; Python reads that same clock.

- Configure the Pi once: `timedatectl`, NTP when online, `hwclock` sync if you use a discrete RTC chip—details depend on your board.
- **Per stereo pair**, we log **one** wall timestamp (not per-eye “frame leveling” for sync). `monotonic_ns` is included for ordering and intervals without assuming clock steps.
- **`--timestamps-csv path.csv`**: columns `frame_index`, `pair_utc_iso`, `unix_ns`, `monotonic_ns`.
- **`--start-at`**: block until wall time, then open cameras (ISO8601; suffix `Z` for UTC, or naive for local).
- **`--no-pace-fps`**: disable sleeps that try to hit `--fps`; capture as fast as the pipeline allows while timestamps record **when** each pair was taken.
- **`--session-json`**: writes `{prefix}_session.json` with start/end UTC and pair count.

For recurring jobs (e.g. every day at 06:00), prefer **systemd timer** or **cron** launching this script at that time, instead of a long-running wait.

## Stereo capture caveat

Left and right images in a pair are still grabbed sequentially; wall-clock logs tell you **when** the pair was completed, not hardware genlock between sensors.

## CLI overrides

Any field can be overridden without editing JSON, e.g. `--width 1280 --camera-right 1 --rig ...`.
