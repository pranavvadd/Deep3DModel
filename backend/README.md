# Backend API

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## Run

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /api/health`
- `GET /api/config`
- `POST /api/convert` (multipart form: `file`, optional `inv=true`)
- `GET /api/downloads/<file>`

## Swap model later

Set `MODEL_PATH` to the new model before starting backend:

```bash
MODEL_PATH=Deep3D/export/your_finetuned_model.pt uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```
