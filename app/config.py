"""Configuration for the CIMPLE factors server."""

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings."""

    # Model configuration
    models_path: Path = Field(
        default=Path("models"),
        description="Path to directory containing BERT model files",
    )
    device: str = Field(
        default="auto", description="PyTorch device (auto, cpu, cuda, cuda:0, etc.)"
    )
    batch_size: int = Field(
        default=32, description="Default batch size for processing", ge=1, le=128
    )
    max_length: int = Field(
        default=128, description="Default maximum sequence length", ge=1, le=512
    )
    frugal_batch_size: int = Field(
        default=8, description="Batch size for the frugal-ai-stance model", ge=1, le=128
    )
    frugal_max_length: int = Field(
        default=256,
        description="Maximum sequence length for the frugal-ai-stance model",
        ge=1,
        le=512,
    )
    auto_download: bool = Field(
        default=True, description="Whether to automatically download missing models"
    )

    # Server configuration
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port", ge=1, le=65535)
    workers: int = Field(default=1, description="Number of worker processes", ge=1)

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    class Config:
        env_prefix = "BERT_"


def get_settings() -> Settings:
    """Get application settings from environment variables."""
    return Settings(
        models_path=Path(os.getenv("BERT_MODELS_PATH", "models")),
        device=os.getenv("BERT_DEVICE", "auto"),
        batch_size=int(os.getenv("BERT_BATCH_SIZE", "32")),
        max_length=int(os.getenv("BERT_MAX_LENGTH", "128")),
        frugal_batch_size=int(os.getenv("BERT_FRUGAL_BATCH_SIZE", "8")),
        frugal_max_length=int(os.getenv("BERT_FRUGAL_MAX_LENGTH", "256")),
        auto_download=os.getenv("BERT_AUTO_DOWNLOAD", "true").lower() == "true",
        host=os.getenv("BERT_HOST", "127.0.0.1"),
        port=int(os.getenv("BERT_PORT", "8000")),
        workers=int(os.getenv("BERT_WORKERS", "1")),
        log_level=os.getenv("BERT_LOG_LEVEL", "INFO"),
    )
