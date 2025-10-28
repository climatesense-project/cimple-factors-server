"""FastAPI application for CIMPLE factors prediction."""

from contextlib import asynccontextmanager
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app import __version__

from .config import get_settings
from .models import BertFactorsPredictor
from .schemas import (
    ConspiracyResult,
    ErrorResponse,
    FactorResult,
    HealthResponse,
    ModelsInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

# Global predictor instance
predictor: BertFactorsPredictor | None = None
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global predictor

    logger.info("Starting server...")
    logger.info(f"Models path: {settings.models_path}")
    logger.info(f"Device: {settings.device}")
    logger.info(f"Batch size: {settings.batch_size}")

    # Initialize predictor
    try:
        predictor = BertFactorsPredictor(
            models_path=settings.models_path,
            device=settings.device,
            batch_size=settings.batch_size,
            max_length=settings.max_length,
            auto_download=settings.auto_download,
        )

        # Initialize models in background
        logger.info("Initializing BERT models...")
        start_time = time.time()
        predictor.initialize()
        load_time = time.time() - start_time
        logger.info(f"Models initialized in {load_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Failed to initialize BERT predictor: {e}")
        predictor = None

    yield

    logger.info("Shutting down server...")


app = FastAPI(
    title="CIMPLE Factors API",
    description=(
        "API for predicting emotion, sentiment, political leaning, tropes, and "
        "conspiracy factors using BERT models"
    ),
    version=__version__,
    lifespan=lifespan,
)


def _convert_to_factor_result(result: dict[str, Any] | None) -> FactorResult | None:
    """Convert prediction result to FactorResult schema."""
    if result is None:
        return None

    conspiracies = result.get("conspiracies", {})
    return FactorResult(
        emotion=result.get("emotion"),
        sentiment=result.get("sentiment"),
        political_leaning=result.get("political_leaning"),
        tropes=result.get("tropes", []),
        persuasion_techniques=result.get("persuasion_techniques", []),
        conspiracies=ConspiracyResult(
            mentioned=conspiracies.get("mentioned", []),
            promoted=conspiracies.get("promoted", []),
        ),
        climate_related=result.get("climate_related"),
    )


@app.get("/", response_model=dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "CIMPLE Factors API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    global predictor

    if predictor is None:
        return HealthResponse(
            status="unhealthy",
            models_loaded=False,
            device="unknown",
            available_models=[],
            missing_models=[],
        )

    missing_models = predictor.get_missing_models()
    available_models = [
        model for model in predictor.REQUIRED_MODELS if model not in missing_models
    ]

    return HealthResponse(
        status="healthy" if predictor.models_loaded else "initializing",
        models_loaded=predictor.models_loaded,
        device=str(predictor.torch_device) if predictor.torch_device else "unknown",
        available_models=available_models,
        missing_models=missing_models,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict factors for a batch of texts."""
    global predictor

    if predictor is None:
        raise HTTPException(status_code=503, detail="BERT predictor not initialized")

    if not predictor.models_loaded:
        raise HTTPException(status_code=503, detail="BERT models not loaded")

    try:
        # Override batch size and max_length if provided
        original_batch_size = predictor.batch_size
        original_max_length = predictor.max_length

        if request.batch_size is not None:
            predictor.batch_size = request.batch_size
        if request.max_length is not None:
            predictor.max_length = request.max_length

        logger.info(f"Processing {len(request.texts)} texts")
        start_time = time.time()

        # Run prediction
        results = predictor.predict(request.texts)

        # Restore original settings
        predictor.batch_size = original_batch_size
        predictor.max_length = original_max_length

        processing_time = time.time() - start_time
        processed_count = sum(1 for r in results if r is not None)

        logger.info(
            f"Processed {processed_count}/{len(request.texts)} texts in {processing_time:.2f}s"
        )

        # Convert results to response schema
        factor_results = [_convert_to_factor_result(result) for result in results]

        return PredictionResponse(
            results=factor_results,
            processed_count=processed_count,
            total_count=len(request.texts),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}") from e


@app.get("/models", response_model=ModelsInfoResponse)
async def models_info():
    """Get information about available models."""
    global predictor

    if predictor is None:
        raise HTTPException(status_code=503, detail="BERT predictor not initialized")

    missing_models = predictor.get_missing_models()
    available_models = [
        model for model in predictor.REQUIRED_MODELS if model not in missing_models
    ]
    return ModelsInfoResponse(
        required_models=predictor.REQUIRED_MODELS,
        available_models=available_models,
        missing_models=missing_models,
        models_path=str(predictor.models_path),
        emotions=predictor.EMOTIONS_LIST,
        sentiments=predictor.SENTIMENTS_LIST,
        political_bias=predictor.POLITICAL_BIAS_LIST,
        conspiracies=predictor.CONSPIRACIES_LIST,
        conspiracy_levels=predictor.CONSPIRACY_LEVELS_LIST,
        tropes=predictor.TROPES_LIST,
        persuasion_techniques=predictor.PERSUASION_TECHNIQUES_LIST,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=None,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
    )
