"""FastAPI application for phishing URL prediction."""

from __future__ import annotations

import sys
import time
from typing import List, Optional

sys.dont_write_bytecode = True

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

try:
    from .model import batch_predict, get_model_metadata, predict_url
except ImportError:  # pragma: no cover - allows running from inside backend/
    from model import batch_predict, get_model_metadata, predict_url


app = FastAPI(
    title="Phishing URL Detection API",
    description="Calibrated phishing URL detection service with cyber risk scoring.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """Input schema for single-URL prediction."""

    url: str = Field(..., min_length=1, max_length=2048, examples=["https://example.com"])

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("URL must not be empty.")
        return stripped


class BatchPredictRequest(BaseModel):
    """Input schema for batch prediction."""

    urls: List[str] = Field(..., min_length=1, max_length=100)

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, value: List[str]) -> List[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty URL is required.")
        return cleaned


class FeatureContribution(BaseModel):
    """Single top-factor explanation entry."""

    feature: str
    feature_value: float
    contribution: float
    description: str


class ReasonDetail(BaseModel):
    """Structured heuristic explanation entry."""

    reason: str
    weight: float
    signal: str


class PredictResponse(BaseModel):
    """Prediction response schema."""

    url: str
    prediction: str
    risk_band: str
    risk_score: float
    risk_level: str
    confidence: str
    confidence_score: float
    probability: float
    base_model_probability: float
    heuristic_probability: float
    risk_score_sigmoid: float
    model_threshold: float
    effective_phishing_min_probability: float
    risk_bands: dict
    hard_rule_phishing: bool
    dual_threshold_score: float
    dual_threshold_label: str
    dual_threshold_confidence: str
    top_risk_factors: List[str]
    reasons: List[str] = Field(default_factory=list)
    reason_details: List[ReasonDetail] = Field(default_factory=list)
    feature_contributions: List[FeatureContribution]
    extracted_features: dict
    model_name: str
    processing_time_ms: Optional[float] = None


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect the root route to the API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"])
async def health() -> dict:
    """Return model metadata and service health."""
    try:
        return {
            "status": "healthy",
            "model": get_model_metadata(),
            "api_version": app.version,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-url", response_model=PredictResponse, tags=["Prediction"])
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_endpoint(request: PredictRequest) -> dict:
    """Predict phishing risk for a single URL."""
    try:
        start = time.perf_counter()
        result = predict_url(request.url)
        result["processing_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/batch", tags=["Prediction"])
async def batch_endpoint(request: BatchPredictRequest) -> dict:
    """Predict phishing risk for a batch of URLs."""
    try:
        start = time.perf_counter()
        results = batch_predict(request.urls)
        return {
            "count": len(results),
            "results": results,
            "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc
