# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportCallIssue=false, reportPossiblyUnboundVariable=false
"""BERT model classes for factors prediction."""

import logging
from pathlib import Path
from typing import Any
import warnings

import requests
from tqdm import tqdm

# Suppress specific huggingface-hub deprecation warning
warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*deprecated.*",
    category=FutureWarning,
    module="huggingface_hub",
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    BertForPreTraining,
)

logger = logging.getLogger(__name__)


class CovidTwitterBertClassifier(nn.Module):
    """BERT classifier for COVID Twitter data."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.bert = BertForPreTraining.from_pretrained(  # ty: ignore[possibly-unbound-attribute]
            "digitalepidemiologylab/covid-twitter-bert-v2",
            revision="b113bc3c2590d7b32ed62603fe1ebe32e1e5beee",
        )
        original_in_features = self.bert.cls.seq_relationship.in_features
        self.bert.cls.seq_relationship = nn.Linear(original_in_features, n_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
        )
        logits: torch.Tensor = outputs[1]
        return logits


class BertFactorsPredictor:
    """BERT-based factors predictor for emotion, sentiment, political leaning, and conspiracy detection."""

    # Constants for factor categories
    EMOTIONS_LIST = ["None", "Happiness", "Anger", "Sadness", "Fear"]
    POLITICAL_BIAS_LIST = ["Left", "Other", "Right"]
    SENTIMENTS_LIST = ["Negative", "Neutral", "Positive"]
    CONSPIRACIES_LIST = [
        "Suppressed Cures",
        "Behaviour and mind Control",
        "Antivax",
        "Fake virus",
        "Intentional Pandemic",
        "Harmful Radiation",
        "Population Reduction",
        "New World Order",
        "Satanism",
    ]
    CONSPIRACY_LEVELS_LIST = ["No ", "Mentioning ", "Supporting "]

    # Model download configuration
    MODELS_BASE_URL = "https://data.cimple.eu/models"
    REQUIRED_MODELS = [
        "emotion.pth",
        "sentiment.pth",
        "political-leaning.pth",
        "conspiracy.pth",
    ]

    def __init__(
        self,
        models_path: str | Path,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 128,
        auto_download: bool = True,
    ):
        self.models_path = Path(models_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.auto_download = auto_download

        self.tokenizer: AutoTokenizer | None = None
        self.models: (
            tuple[
                CovidTwitterBertClassifier | None,
                CovidTwitterBertClassifier | None,
                CovidTwitterBertClassifier | None,
                CovidTwitterBertClassifier | None,
            ]
            | None
        ) = None
        self.torch_device: torch.device | None = None
        self.models_loaded = False

    def _setup_device(self) -> None:
        """Setup PyTorch device."""
        if self.device == "auto":
            self.torch_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.torch_device = torch.device(self.device)

        logger.info(f"Using PyTorch device: {self.torch_device}")

    def ensure_models_available(self) -> None:
        """Ensure all required models are available, download if missing and auto_download is enabled."""
        if not self.auto_download:
            return

        missing_models = self.get_missing_models()
        if missing_models:
            logger.info(f"Missing BERT models: {missing_models}")
            logger.info("Downloading missing models...")

            self.models_path.mkdir(parents=True, exist_ok=True)

            for model_file in missing_models:
                self._download_model(model_file)
        else:
            logger.info("All BERT models are available")

    def get_missing_models(self) -> list[str]:
        """Get list of missing model files."""
        missing = []
        for model_file in self.REQUIRED_MODELS:
            model_path = self.models_path / model_file
            if not model_path.exists():
                missing.append(model_file)
        return missing

    def _download_model(self, model_file: str) -> None:
        """Download a single model file."""
        url = f"{self.MODELS_BASE_URL}/{model_file}"
        model_path = self.models_path / model_file

        logger.info(f"Downloading {model_file} from {url}")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(model_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=model_file,
                    miniters=1,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Successfully downloaded {model_file}")

        except Exception as e:
            logger.error(f"Failed to download {model_file}: {e}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Could not download required model {model_file}") from e

    def _load_models(self) -> None:
        """Load BERT models and tokenizer."""
        if not self.models_path.exists():
            logger.error(f"Models path does not exist: {self.models_path}")
            return

        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "digitalepidemiologylab/covid-twitter-bert-v2",
                revision="b113bc3c2590d7b32ed62603fe1ebe32e1e5beee",
            )

            # Load models
            logger.info("Loading BERT factor models...")

            # Emotion model
            model_em: CovidTwitterBertClassifier | None = None
            try:
                model_em = CovidTwitterBertClassifier(len(self.EMOTIONS_LIST)).to(
                    self.torch_device
                )
                emotion_path = self.models_path / "emotion.pth"
                if emotion_path.exists():
                    model_em.load_state_dict(
                        torch.load(
                            emotion_path,
                            map_location=self.torch_device,
                            weights_only=True,
                        )
                    )
                    model_em.eval()
                else:
                    logger.warning(f"Emotion model not found at {emotion_path}")
                    model_em = None
            except Exception as e:
                logger.warning(f"Failed to load emotion model: {e}")
                model_em = None

            # Political leaning model
            model_pol: CovidTwitterBertClassifier | None = None
            try:
                model_pol = CovidTwitterBertClassifier(
                    len(self.POLITICAL_BIAS_LIST)
                ).to(self.torch_device)
                political_path = self.models_path / "political-leaning.pth"
                if political_path.exists():
                    model_pol.load_state_dict(
                        torch.load(
                            political_path,
                            map_location=self.torch_device,
                            weights_only=True,
                        )
                    )
                    model_pol.eval()
                else:
                    logger.warning(
                        f"Political leaning model not found at {political_path}"
                    )
                    model_pol = None
            except Exception as e:
                logger.warning(f"Failed to load political leaning model: {e}")
                model_pol = None

            # Sentiment model
            model_sent: CovidTwitterBertClassifier | None = None
            try:
                model_sent = CovidTwitterBertClassifier(len(self.SENTIMENTS_LIST)).to(
                    self.torch_device
                )
                sentiment_path = self.models_path / "sentiment.pth"
                if sentiment_path.exists():
                    model_sent.load_state_dict(
                        torch.load(
                            sentiment_path,
                            map_location=self.torch_device,
                            weights_only=True,
                        )
                    )
                    model_sent.eval()
                else:
                    logger.warning(f"Sentiment model not found at {sentiment_path}")
                    model_sent = None
            except Exception as e:
                logger.warning(f"Failed to load sentiment model: {e}")
                model_sent = None

            # Conspiracy model
            model_con: CovidTwitterBertClassifier | None = None
            try:
                model_con = CovidTwitterBertClassifier(
                    len(self.CONSPIRACIES_LIST) * len(self.CONSPIRACY_LEVELS_LIST)
                ).to(self.torch_device)
                conspiracy_path = self.models_path / "conspiracy.pth"
                if conspiracy_path.exists():
                    model_con.load_state_dict(
                        torch.load(
                            conspiracy_path,
                            map_location=self.torch_device,
                            weights_only=True,
                        )
                    )
                    model_con.eval()
                else:
                    logger.warning(f"Conspiracy model not found at {conspiracy_path}")
                    model_con = None
            except Exception as e:
                logger.warning(f"Failed to load conspiracy model: {e}")
                model_con = None

            self.models = (model_em, model_pol, model_sent, model_con)

        except Exception as e:
            logger.error(f"Error loading BERT models: {e}")
            self.models = None
            self.tokenizer = None

    def initialize(self) -> None:
        """Initialize the predictor by loading models."""
        if not self.models_loaded:
            self._setup_device()
            self.ensure_models_available()
            self._load_models()
            self.models_loaded = True

    def is_available(self) -> bool:
        """Check if BERT predictor is available."""
        if not self.auto_download:
            missing_models = self._get_missing_models()
            if missing_models:
                logger.warning(f"Missing BERT models: {missing_models}")
                return False
        return True

    def predict(self, texts: list[str]) -> list[dict[str, Any] | None]:
        """
        Predict factors for a list of texts using batched processing.

        Args:
            texts: List of texts to analyze

        Returns:
            List of predicted factors for each text
        """
        if not texts:
            return []

        if not self.models_loaded:
            raise RuntimeError("Models not initialized. Call initialize() first.")

        if self.tokenizer is None or self.models is None:
            logger.error("Tokenizer or models not loaded")
            return [None] * len(texts)

        try:
            # Filter out empty texts
            valid_items = [
                (i, text) for i, text in enumerate(texts) if text and text.strip()
            ]
            valid_indices, valid_texts = (
                zip(*valid_items, strict=False) if valid_items else ([], [])
            )
            valid_indices, valid_texts = list(valid_indices), list(valid_texts)

            if not valid_texts:
                return [None] * len(texts)

            model_em, model_pol, model_sent, model_con = self.models

            # Initialize result lists
            all_predictions_em = []
            all_predictions_pol = []
            all_predictions_sent = []
            all_predictions_con = []

            # Process texts in batches
            num_batches = (len(valid_texts) + self.batch_size - 1) // self.batch_size
            logger.info(
                f"Processing {len(valid_texts)} texts in {num_batches} batches of size {self.batch_size}"
            )

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(valid_texts))
                batch_texts = valid_texts[start_idx:end_idx]

                if batch_idx % 10 == 0:
                    logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")

                # Batch tokenization
                tokenized_batch = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = tokenized_batch["input_ids"].to(self.torch_device)
                token_type_ids = tokenized_batch["token_type_ids"].to(self.torch_device)
                attention_mask = tokenized_batch["attention_mask"].to(self.torch_device)

                with torch.no_grad():
                    # Batch predictions for this batch
                    if model_em:
                        logits_em_batch = model_em(
                            input_ids, token_type_ids, attention_mask
                        )
                        predictions_em = (
                            logits_em_batch.detach().cpu().numpy().argmax(axis=1)
                        )
                        all_predictions_em.extend(predictions_em)

                    if model_pol:
                        logits_pol_batch = model_pol(
                            input_ids, token_type_ids, attention_mask
                        )
                        predictions_pol = (
                            logits_pol_batch.detach().cpu().numpy().argmax(axis=1)
                        )
                        all_predictions_pol.extend(predictions_pol)

                    if model_sent:
                        logits_sent_batch = model_sent(
                            input_ids, token_type_ids, attention_mask
                        )
                        predictions_sent = (
                            logits_sent_batch.detach().cpu().numpy().argmax(axis=1)
                        )
                        all_predictions_sent.extend(predictions_sent)

                    if model_con:
                        logits_con_batch = model_con(
                            input_ids, token_type_ids, attention_mask
                        )

                        num_conspiracies = len(self.CONSPIRACIES_LIST)
                        num_levels = len(self.CONSPIRACY_LEVELS_LIST)

                        predictions_con_reshaped = (
                            logits_con_batch.detach()
                            .cpu()
                            .numpy()
                            .reshape(-1, num_conspiracies, num_levels)
                        )
                        predictions_con = predictions_con_reshaped.argmax(axis=2)
                        all_predictions_con.extend(predictions_con)

            # Process results for each valid text
            batch_results: list[dict[str, Any]] = []
            for i in range(len(valid_texts)):
                results: dict[str, Any] = {}

                if all_predictions_em:
                    emotion = self.EMOTIONS_LIST[all_predictions_em[i]]
                    results["emotion"] = emotion if emotion != "None" else None

                if all_predictions_pol:
                    results["political_leaning"] = self.POLITICAL_BIAS_LIST[
                        all_predictions_pol[i]
                    ]

                if all_predictions_sent:
                    results["sentiment"] = self.SENTIMENTS_LIST[all_predictions_sent[i]]

                if all_predictions_con:
                    conspiracy_indices = all_predictions_con[i]
                    mentioned = []
                    promoted = []

                    for j, level_idx in enumerate(conspiracy_indices):
                        conspiracy_name = self.CONSPIRACIES_LIST[j]
                        if level_idx == 1:
                            mentioned.append(conspiracy_name)
                        elif level_idx == 2:
                            promoted.append(conspiracy_name)

                    results["conspiracies"] = {
                        "mentioned": mentioned,
                        "promoted": promoted,
                    }

                batch_results.append(results)

            # Map results back to original indices
            final_results: list[dict[str, Any] | None] = [None] * len(texts)
            for i, original_idx in enumerate(valid_indices):
                final_results[original_idx] = batch_results[i]

            return final_results

        except Exception as e:
            logger.error(f"Error in CIMPLE factors prediction: {e}")
            return [None] * len(texts)
