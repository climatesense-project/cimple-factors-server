"""Pydantic schemas for the CIMPLE factors API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for batch prediction."""

    texts: list[str] = Field(..., description="List of texts to analyze", min_length=1)
    batch_size: int | None = Field(
        None, description="Batch size for processing", ge=1, le=128
    )
    max_length: int | None = Field(
        None, description="Maximum sequence length", ge=1, le=512
    )


class ConspiracyResult(BaseModel):
    """Conspiracy detection result."""

    mentioned: list[str] = Field(
        default_factory=list, description="Mentioned conspiracy theories"
    )
    promoted: list[str] = Field(
        default_factory=list, description="Promoted conspiracy theories"
    )


class FactorResult(BaseModel):
    """Individual factor prediction result."""

    emotion: str | None = Field(None, description="Predicted emotion")
    sentiment: str | None = Field(None, description="Predicted sentiment")
    political_leaning: str | None = Field(
        None, description="Predicted political leaning"
    )
    tropes: list[str] = Field(
        default_factory=list, description="Detected narrative tropes"
    )
    persuasion_techniques: list[str] = Field(
        default_factory=list, description="Detected persuasion techniques"
    )
    conspiracies: ConspiracyResult = Field(
        default_factory=ConspiracyResult, description="Conspiracy predictions"
    )
    climate_related: bool | None = Field(
        None, description="Whether the text is climate-related"
    )


class PredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    results: list[FactorResult | None] = Field(
        ..., description="Prediction results for each text"
    )
    processed_count: int = Field(
        ..., description="Number of texts successfully processed"
    )
    total_count: int = Field(..., description="Total number of texts in request")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    device: str = Field(..., description="PyTorch device being used")
    available_models: list[str] = Field(..., description="List of available models")
    missing_models: list[str] = Field(..., description="List of missing models")


class ModelsInfoResponse(BaseModel):
    """Response schema for models information."""

    required_models: list[str] = Field(..., description="List of required models")
    available_models: list[str] = Field(..., description="List of available models")
    missing_models: list[str] = Field(..., description="List of missing models")
    models_path: str = Field(..., description="Path to models directory")
    emotions: list[str] = Field(..., description="List of possible emotions")
    sentiments: list[str] = Field(..., description="List of possible sentiments")
    political_bias: list[str] = Field(
        ..., description="List of possible political leanings"
    )
    conspiracies: list[str] = Field(..., description="List of conspiracy theories")
    conspiracy_levels: list[str] = Field(..., description="List of conspiracy levels")
    tropes: list[str] = Field(..., description="List of narrative tropes")
    persuasion_techniques: list[str] = Field(
        ..., description="List of persuasion techniques"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")
