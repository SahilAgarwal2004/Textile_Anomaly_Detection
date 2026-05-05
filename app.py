"""
TEXTILE ANOMALY DETECTION — FastAPI Server
==========================================
Wraps main.py into a REST API.

Endpoints:
  POST /predict        — Upload an image, get prediction + optional defect info
  GET  /health         — Health check (models loaded?)
  GET  /               — API info

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then call:
    curl -X POST "http://localhost:8000/predict" \
         -F "file=@ITD/type3cam1/test/anomaly/5235.png"
"""

import io
import os
import tempfile
import traceback

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# ── Import your inference engine ──────────────────────────────
import main as inference_engine

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Textile Anomaly Detection API",
    description=(
        "Upload a fabric image to detect defects. "
        "Returns prediction (NORMAL / ANOMALY), a text reasoning, "
        "and a base64-encoded defect patch image when anomaly is found."
    ),
    version="1.0.0",
)

# ─────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    status: str                      # "ok" or "error"
    prediction: Optional[str]        # "NORMAL" or "ANOMALY"
    reasoning: Optional[str]         # Human-readable defect explanation
    defect_patch_base64: Optional[str]  # PNG image as base64 string (None if NORMAL)
    message: Optional[str]           # Only present on error


class HealthResponse(BaseModel):
    status: str        # "ready" or "loading"
    models_loaded: bool


# ─────────────────────────────────────────────────────────────
# Load models once at startup
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_models_on_startup():
    print("Loading models at startup...")
    try:
        inference_engine.load_models()
        print("All models ready.")
    except Exception as e:
        print(f"WARNING: Model loading failed at startup: {e}")
        print("Models will be lazy-loaded on first request.")


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    """API info and usage."""
    return {
        "name": "Textile Anomaly Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Upload an image to detect fabric defects",
            "GET  /health":  "Check if models are loaded and ready",
        },
        "usage": {
            "curl": (
                'curl -X POST "http://localhost:8000/predict" '
                '-F "file=@your_image.png"'
            ),
            "python": (
                "import requests\n"
                "with open('image.png', 'rb') as f:\n"
                "    r = requests.post('http://localhost:8000/predict', files={'file': f})\n"
                "print(r.json())"
            ),
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns whether models are loaded and ready to serve predictions."""
    loaded = inference_engine._unet is not None
    return HealthResponse(
        status="ready" if loaded else "loading",
        models_loaded=loaded,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a fabric image to run anomaly detection.

    - **file**: Image file (PNG, JPG, JPEG, BMP)

    Returns:
    - **prediction**: "NORMAL" or "ANOMALY"
    - **reasoning**: Text description of the defect (only for ANOMALY)
    - **defect_patch_base64**: Base64-encoded PNG of the defect region (only for ANOMALY)
    """

    # ── Validate file type ────────────────────────────────────
    allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed_extensions)}",
        )

    # ── Ensure models are loaded ──────────────────────────────
    if inference_engine._unet is None:
        try:
            inference_engine.load_models()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model loading failed: {str(e)}",
            )

    # ── Save upload to a temp file (cv2.imread needs a path) ──
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {e}")

    # ── Run inference ─────────────────────────────────────────
    try:
        result = inference_engine.run_inference(tmp_path)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}",
        )
    finally:
        # Always clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # ── Build response ────────────────────────────────────────
    if result["status"] == "error":
        return PredictionResponse(
            status="error",
            prediction=None,
            reasoning=None,
            defect_patch_base64=None,
            message=result.get("message", "Unknown error"),
        )

    return PredictionResponse(
        status="ok",
        prediction=result["prediction"],
        reasoning=result.get("reasoning"),
        defect_patch_base64=result.get("patch_b64"),
        message=None,
    )


# ─────────────────────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)