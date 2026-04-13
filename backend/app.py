"""
app.py — FastAPI Application for Phishing URL Detection
========================================================
Run with:
    uvicorn backend.app:app --reload --port 8000

Endpoints
---------
    GET  /              → redirect to docs
    GET  /health        → model metadata + status
    POST /predict       → single URL prediction
    POST /batch         → batch URL prediction (max 100)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import time

from backend.model import predict_url, batch_predict, get_model_metadata

# ── App initialisation ────────────────────────────────────────────────────────
app = FastAPI(
    title="Phishing URL Detection API",
    description=(
        "Production-grade ML API that detects malicious / phishing URLs. "
        "Built on Random Forest trained on 549K+ real-world URLs."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow all origins for development; restrict in production ───────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        example="https://secure-paypal-login.verify-account.xyz",
    )

    @validator("url")
    def url_not_empty(cls, v):
        if not v.strip():
            raise ValueError("URL must not be empty or only whitespace.")
        return v.strip()


class BatchPredictRequest(BaseModel):
    urls: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of URLs to analyse (max 100 per request).",
    )

    @validator("urls")
    def urls_not_empty(cls, v):
        if not v:
            raise ValueError("URL list must not be empty.")
        return [u.strip() for u in v if u.strip()]


class TopReason(BaseModel):
    feature: str
    value: float
    contribution: float
    description: str


class PredictResponse(BaseModel):
    url: str
    prediction: str                  # "Safe" | "Malicious" | "Error"
    risk_score: float                # 0–100
    confidence: float                # 0–1
    top_reasons: List[TopReason]
    risk_level: str                  # "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
    is_trusted: bool
    is_short_url: bool
    is_ip_url: bool
    warning: Optional[str]
    model_version: str
    threshold: float
    processing_time_ms: Optional[float] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"])
async def health():
    """
    Health check endpoint. Returns model metadata and system status.
    """
    try:
        meta = get_model_metadata()
        return {
            "status": "healthy",
            "model": meta,
            "api_version": "2.0.0",
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {str(e)}. Run 'python train.py' first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Analyse a single URL and return a phishing risk assessment.

    - **url**: The URL to analyse (with or without scheme prefix).

    Returns a detailed risk assessment including:
    - `prediction`: "Safe" or "Malicious"
    - `risk_score`: 0–100 (higher = more suspicious)
    - `risk_level`: LOW / MEDIUM / HIGH
    - `top_reasons`: Top 5 features driving the prediction
    - `confidence`: Raw model probability
    - `is_trusted`: True if domain is in known-safe whitelist
    - `is_short_url`: True if URL uses a shortener (can't inspect destination)
    - `is_ip_url`: True if host is a raw IP address (suspicious)
    """
    try:
        t0 = time.perf_counter()
        result = predict_url(request.url)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        result["processing_time_ms"] = elapsed_ms
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch", tags=["Prediction"])
async def batch(request: BatchPredictRequest):
    """
    Analyse up to 100 URLs in a single request.

    Returns a list of risk assessments, one per URL.
    """
    try:
        t0 = time.perf_counter()
        results = batch_predict(request.urls)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "results": results,
            "count": len(results),
            "processing_time_ms": elapsed_ms,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


try:
    from backend.app_runtime import app  # noqa: E402,F401
except ModuleNotFoundError as exc:
    if exc.name != "backend":
        raise
    from app_runtime import app  # type: ignore[no-redef]  # noqa: E402,F401
