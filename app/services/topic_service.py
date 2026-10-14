from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from app.core.config import get_settings
from app.core.exceptions import (
    ConfigurationError,
    InvalidTextError,
    ModelInferenceError,
    ModelLoadError,
)
from app.schemas.analysis import TopicResult


class TopicService:
    """
    Municipal topic classification service.

    Loads the trained local topic classification model once
    and provides topic prediction for individual texts and batches.

    The model predicts one of the configured municipal topics.

    Example topics:
        roads_infrastructure
        water
        education_and_exams
        transportation_and_smart_city
        public_leadership_and_praise
        religion_and_spirituality
        ...
    """

    _instance: "TopicService | None" = None

    def __init__(self) -> None:
        settings = get_settings()

        self.settings = settings

        try:
            self.device = self._resolve_device(
                settings.MODEL_DEVICE,
            )

            self.model_path = self._resolve_model_path(
                settings.TOPIC_MODEL_PATH,
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

            self.id_to_topic = self._build_id_to_topic()

            self._validate_topic_mapping()

        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            raise ModelLoadError(
                f"Failed to load topic model: {exc}"
            ) from exc

    @classmethod
    def get_instance(cls) -> "TopicService":
        """
        Return the shared topic service instance.

        The model is loaded only once per application process.
        """

        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    def predict(self, text: str) -> TopicResult:
        """
        Predict topic for a single text.
        """

        if not text or not text.strip():
            raise InvalidTextError(
                "Text must not be empty."
            )

        return self.predict_batch([text])[0]

    def predict_batch(
        self,
        texts: list[str],
    ) -> list[TopicResult]:
        """
        Predict topics for multiple texts in one inference call.
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

            results: list[TopicResult] = []

            for prediction, probability in zip(
                predictions,
                probabilities,
            ):
                topic_id = int(prediction.item())

                confidence = float(
                    probability[topic_id].item()
                )

                topic = self.id_to_topic.get(
                    topic_id,
                    str(topic_id),
                )

                results.append(
                    TopicResult(
                        topic=topic,
                        topic_id=topic_id,
                        confidence=confidence,
                    )
                )

            return results

        except Exception as exc:
            raise ModelInferenceError(
                "Topic model inference failed."
            ) from exc

    def _build_id_to_topic(self) -> dict[int, str]:
        """
        Build numeric topic ID → topic name mapping.

        The model configuration is the primary source.

        Example:

            0 → roads_infrastructure
            1 → water
            2 → education_and_exams
            ...
        """

        id2label: dict[Any, Any] = getattr(
            self.model.config,
            "id2label",
            {},
        )

        mapping: dict[int, str] = {}

        for key, value in id2label.items():
            try:
                topic_id = int(key)
            except (TypeError, ValueError):
                continue

            mapping[topic_id] = self._normalize_topic(
                str(value),
            )

        return mapping

    @staticmethod
    def _normalize_topic(topic: str) -> str:
        """
        Normalize model topic labels to the API taxonomy format.
        """

        return (
            topic.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    def _validate_topic_mapping(self) -> None:
        """
        Validate that the model contains a usable topic mapping.
        """

        if not self.id_to_topic:
            raise ConfigurationError(
                "Topic model does not contain a valid id2label mapping."
            )

        expected_num_topics = getattr(
            self.settings,
            "TOPIC_NUM_LABELS",
            None,
        )

        if (
            expected_num_topics is not None
            and len(self.id_to_topic) != expected_num_topics
        ):
            raise ConfigurationError(
                "Topic model label count does not match "
                f"TOPIC_NUM_LABELS={expected_num_topics}. "
                f"Found {len(self.id_to_topic)} labels."
            )

        topic_names = list(self.id_to_topic.values())

        if len(topic_names) != len(set(topic_names)):
            raise ConfigurationError(
                "Topic model contains duplicate topic labels."
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
        Resolve and validate the local topic model directory.
        """

        path = Path(configured_path)

        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            raise FileNotFoundError(
                f"Topic model directory not found: {path}"
            )

        if not (path / "config.json").exists():
            raise FileNotFoundError(
                f"Topic model config.json not found: {path}"
            )

        if not (
            (path / "model.safetensors").exists()
            or (path / "pytorch_model.bin").exists()
        ):
            raise FileNotFoundError(
                f"Topic model weights not found: {path}"
            )

        return str(path)