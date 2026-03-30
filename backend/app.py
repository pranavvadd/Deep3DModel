import os
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parents[1]
DEEP3D_DIR = ROOT_DIR / "Deep3D"
INFERENCE_SCRIPT = DEEP3D_DIR / "inference.py"
UPLOAD_DIR = ROOT_DIR / "backend" / "uploads"
OUTPUT_DIR = ROOT_DIR / "backend" / "outputs"

# Switch models by changing MODEL_PATH env var. Example:
# MODEL_PATH=Deep3D/export/deep3d_v1.0_640x360_cpu.pt
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEEP3D_DIR / "export" / "deep3d_v1.0_640x360_cpu.pt")))
INFERENCE_TIMEOUT_SECONDS = int(os.getenv("INFERENCE_TIMEOUT_SECONDS", "7200"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Deep3D API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/downloads", StaticFiles(directory=str(OUTPUT_DIR)), name="downloads")


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict:
    return {
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "inference_timeout_seconds": INFERENCE_TIMEOUT_SECONDS,
    }


@app.post("/api/convert")
async def convert_video(
    file: UploadFile = File(...),
    inv: bool = Form(False),
) -> dict:
    if not INFERENCE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="inference.py not found")

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found at {MODEL_PATH}. Set MODEL_PATH environment variable.",
        )

    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    allowed = {".mp4", ".mov", ".avi", ".mkv"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    job_id = uuid.uuid4().hex[:12]
    input_path = UPLOAD_DIR / f"{job_id}{suffix}"
    output_path = OUTPUT_DIR / f"{job_id}_3d.mp4"
    tmp_dir = ROOT_DIR / "backend" / "tmp" / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cmd = [
        "python3",
        str(INFERENCE_SCRIPT),
        "--model",
        str(MODEL_PATH),
        "--video",
        str(input_path),
        "--out",
        str(output_path),
        "--tmpdir",
        str(tmp_dir),
    ]
    if inv:
        cmd.append("--inv")

    try:
        subprocess.run(
            cmd,
            cwd=str(DEEP3D_DIR),
            check=True,
            capture_output=True,
            text=True,
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Inference timed out") from exc
    except subprocess.CalledProcessError as exc:
        error_log = (exc.stderr or exc.stdout or "Inference failed")[-3000:]
        raise HTTPException(status_code=500, detail=error_log) from exc

    if not output_path.exists():
        raise HTTPException(status_code=500, detail="Output file was not generated")

    return {
        "job_id": job_id,
        "output_file": output_path.name,
        "download_url": f"/api/downloads/{output_path.name}",
    }


@app.get("/api/download/{file_name}")
def download_file(file_name: str):
    file_path = OUTPUT_DIR / file_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), filename=file_name, media_type="video/mp4")
