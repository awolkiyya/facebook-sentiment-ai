from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import get_settings
from app.core.exceptions import (
    ConfigurationError,
    InvalidTextError,
    ModelInferenceError,
    ModelLoadError,
)
from app.schemas.analysis import LanguageResult


class LanguageService:
    """
    Language detection service.

    Loads the trained language classification model once and
    provides language prediction for individual texts and batches.
    """

    _instance: "LanguageService | None" = None

    def __init__(self) -> None:
        settings = get_settings()

        self.settings = settings

        try:
            self.device = self._resolve_device(
                settings.MODEL_DEVICE,
            )

            self.model_path = self._resolve_model_path(
                settings.LANGUAGE_MODEL_PATH,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True,
            )

            self.model.to(self.device)
            self.model.eval()

            self.id_to_label = self._build_id_to_label()

        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            raise ModelLoadError(
                f"Failed to load language model: {exc}"
            ) from exc

    @classmethod
    def get_instance(cls) -> "LanguageService":
        """
        Return the shared language service instance.

        The model is loaded only once per application process.
        """

        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    def predict(self, text: str) -> LanguageResult:
        """
        Detect the language of a single text.
        """

        if not text or not text.strip():
            raise InvalidTextError(
                "Text must not be empty."
            )

        return self.predict_batch([text])[0]

    def predict_batch(
        self,
        texts: list[str],
    ) -> list[LanguageResult]:
        """
        Detect languages for multiple texts in one inference call.
        """

        if not texts:
            raise InvalidTextError(
                "Texts must not be empty."
            )

        for text in texts:
            if not text or not text.strip():
                raise InvalidTextError(
                    "Texts must not contain empty values."
                )

        try:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.settings.MAX_TEXT_LENGTH,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=-1,
            )

            results: list[LanguageResult] = []

            for prediction, probability in zip(
                predictions,
                probabilities,
            ):
                label_id = int(prediction.item())

                confidence = float(
                    probability[label_id].item()
                )

                label = self.id_to_label.get(
                    label_id,
                    str(label_id),
                )

                results.append(
                    LanguageResult(
                        code=label,
                        confidence=confidence,
                    )
                )

            return results

        except Exception as exc:
            raise ModelInferenceError(
                "Language model inference failed."
            ) from exc

    def _build_id_to_label(self) -> dict[int, str]:
        """
        Build the numeric-ID to language-code mapping.

        Expected languages:
            0 = om
            1 = am
            2 = en

        The model configuration is the primary source of truth.
        """

        id2label: dict[Any, Any] = getattr(
            self.model.config,
            "id2label",
            {},
        )

        mapping: dict[int, str] = {}

        for key, value in id2label.items():
            try:
                label_id = int(key)
            except (TypeError, ValueError):
                continue

            mapping[label_id] = self._normalize_label(
                str(value),
            )

        if mapping:
            return mapping

        return {
            0: "om",
            1: "am",
            2: "en",
        }

    @staticmethod
    def _normalize_label(label: str) -> str:
        """
        Normalize model labels to API language codes.
        """

        normalized = label.strip().lower()

        aliases = {
            "om": "om",
            "orm": "om",
            "oromo": "om",
            "oromiffa": "om",
            "afaan_oromo": "om",

            "am": "am",
            "amh": "am",
            "amharic": "am",

            "en": "en",
            "eng": "en",
            "english": "en",
        }

        return aliases.get(
            normalized,
            normalized,
        )

    @staticmethod
    def _resolve_device(
        configured_device: str,
    ) -> torch.device:
        """
        Resolve the inference device.

        auto:
            CUDA → MPS → CPU
        """

        device = configured_device.lower().strip()

        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")

            if torch.backends.mps.is_available():
                return torch.device("mps")

            return torch.device("cpu")

        if device == "cuda":
            if not torch.cuda.is_available():
                raise ConfigurationError(
                    "MODEL_DEVICE=cuda but CUDA is not available."
                )

            return torch.device("cuda")

        if device == "mps":
            if not torch.backends.mps.is_available():
                raise ConfigurationError(
                    "MODEL_DEVICE=mps but Apple MPS is not available."
                )

            return torch.device("mps")

        if device == "cpu":
            return torch.device("cpu")

        raise ConfigurationError(
            f"Unsupported MODEL_DEVICE: {configured_device}"
        )

    @staticmethod
    def _resolve_model_path(
        configured_path: str,
    ) -> str:
        """
        Resolve and validate the local model directory.
        """

        path = Path(configured_path)

        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            raise FileNotFoundError(
                f"Language model directory not found: {path}"
            )

        if not (path / "config.json").exists():
            raise FileNotFoundError(
                f"Language model config.json not found: {path}"
            )

        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(
                f"Language model weights not found: {path}"
            )

        return str(path)