from __future__ import annotations

import logging
import time

from app.core.exceptions import (
    InvalidTextError,
    ModelInferenceError,
)
from app.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    BatchAnalyzeResult,
    LanguageResult,
    ModelInfo,
    ModelVersion,
    SentimentResult,
    TopicResult,
)
from app.services.language_service import LanguageService
from app.services.sentiment_service import SentimentService
from app.services.topic_service import TopicService


logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Application-level orchestration service.

    Coordinates:
        - LanguageService
        - SentimentService
        - TopicService

    This service does not load AI models directly.
    It reuses the shared model-service instances.
    """

    _instance: "AnalysisService | None" = None

    def __init__(
        self,
        language_service: LanguageService | None = None,
        sentiment_service: SentimentService | None = None,
        topic_service: TopicService | None = None,
    ) -> None:
        """
        Initialize the analysis service.

        The underlying language, sentiment, and topic models
        are loaded through their shared service instances.
        """

        logger.info(
            "Initializing AnalysisService."
        )

        try:
            # -----------------------------------------------------------------
            # Language Service
            # -----------------------------------------------------------------

            self.language_service = (
                language_service
                or LanguageService.get_instance()
            )

            logger.info(
                "LanguageService initialized successfully."
            )

            # -----------------------------------------------------------------
            # Sentiment Service
            # -----------------------------------------------------------------

            self.sentiment_service = (
                sentiment_service
                or SentimentService.get_instance()
            )

            logger.info(
                "SentimentService initialized successfully."
            )

            # -----------------------------------------------------------------
            # Topic Service
            # -----------------------------------------------------------------

            self.topic_service = (
                topic_service
                or TopicService.get_instance()
            )

            logger.info(
                "TopicService initialized successfully."
            )

            logger.info(
                "AnalysisService initialized successfully."
            )

        except Exception:
            logger.exception(
                "Failed to initialize AnalysisService."
            )
            raise

    @classmethod
    def get_instance(cls) -> "AnalysisService":
        """
        Return the shared analysis service instance.

        The underlying language, sentiment, and topic models
        are also reused rather than loaded for every request.
        """

        if cls._instance is None:
            logger.info(
                "Creating AnalysisService singleton instance."
            )

            cls._instance = cls()

        else:
            logger.debug(
                "Reusing existing AnalysisService singleton."
            )

        return cls._instance

    # =========================================================================
    # Single Analysis
    # =========================================================================

    def analyze(
        self,
        request: AnalyzeRequest,
    ) -> AnalyzeResponse:
        """
        Analyze a single text.

        Pipeline:

            Text
              ↓
            Language detection
              ↓
            Sentiment prediction
              ↓
            Topic prediction
              ↓
            Unified response
        """

        text = self._validate_text(request.text)

        logger.info(
            "Starting text analysis. text_length=%d",
            len(text),
        )

        started_at = time.perf_counter()

        try:
            # -----------------------------------------------------------------
            # Language Detection
            # -----------------------------------------------------------------

            logger.debug(
                "Running language detection."
            )

            language: LanguageResult = (
                self.language_service.predict(text)
            )

            logger.info(
                "Language detection completed. "
                "language=%s confidence=%.4f",
                language.code,
                language.confidence,
            )

            # -----------------------------------------------------------------
            # Sentiment Prediction
            # -----------------------------------------------------------------

            logger.debug(
                "Running sentiment prediction."
            )

            sentiment: SentimentResult = (
                self.sentiment_service.predict(text)
            )

            logger.info(
                "Sentiment prediction completed. "
                "label=%s label_id=%d confidence=%.4f",
                sentiment.label,
                sentiment.label_id,
                sentiment.confidence,
            )

            # -----------------------------------------------------------------
            # Topic Prediction
            # -----------------------------------------------------------------

            logger.debug(
                "Running topic prediction."
            )

            topic: TopicResult = (
                self.topic_service.predict(text)
            )

            logger.info(
                "Topic prediction completed. "
                "topic=%s topic_id=%d confidence=%.4f",
                topic.topic,
                topic.topic_id,
                topic.confidence,
            )

            # -----------------------------------------------------------------
            # Processing Time
            # -----------------------------------------------------------------

            processing_time_ms = self._processing_time_ms(
                started_at
            )

            # -----------------------------------------------------------------
            # Response
            # -----------------------------------------------------------------

            response = AnalyzeResponse(
                text=text,
                language=language,
                sentiment=sentiment,
                topic=topic,
                model=self._model_info(),
                processing_time_ms=processing_time_ms,
            )

            logger.info(
                "Text analysis completed successfully. "
                "processing_time_ms=%d",
                processing_time_ms,
            )

            return response

        except InvalidTextError:
            logger.warning(
                "Invalid text supplied for analysis."
            )
            raise

        except ModelInferenceError:
            logger.exception(
                "Model inference failed during text analysis."
            )
            raise

        except Exception as exc:
            logger.exception(
                "Unexpected error during text analysis. "
                "error_type=%s error=%s",
                type(exc).__name__,
                str(exc),
            )

            raise ModelInferenceError(
                "Failed to analyze text."
            ) from exc

    # =========================================================================
    # Batch Analysis
    # =========================================================================

    def analyze_batch(
        self,
        request: BatchAnalyzeRequest,
    ) -> BatchAnalyzeResponse:
        """
        Analyze multiple texts.

        Language, sentiment, and topic inference are performed
        as batches so transformer models can process multiple
        texts efficiently.

        The original client-provided IDs are preserved.
        """

        if not request.items:
            raise InvalidTextError(
                "Batch must contain at least one item."
            )

        logger.info(
            "Starting batch analysis. batch_size=%d",
            len(request.items),
        )

        texts = [
            self._validate_text(item.text)
            for item in request.items
        ]

        started_at = time.perf_counter()

        try:
            # -----------------------------------------------------------------
            # Language Detection
            # -----------------------------------------------------------------

            logger.debug(
                "Running batch language detection. "
                "batch_size=%d",
                len(texts),
            )

            languages: list[LanguageResult] = (
                self.language_service.predict_batch(texts)
            )

            logger.info(
                "Batch language detection completed. "
                "results=%d",
                len(languages),
            )

            # -----------------------------------------------------------------
            # Sentiment Prediction
            # -----------------------------------------------------------------

            logger.debug(
                "Running batch sentiment prediction. "
                "batch_size=%d",
                len(texts),
            )

            sentiments: list[SentimentResult] = (
                self.sentiment_service.predict_batch(texts)
            )

            logger.info(
                "Batch sentiment prediction completed. "
                "results=%d",
                len(sentiments),
            )

            # -----------------------------------------------------------------
            # Topic Prediction
            # -----------------------------------------------------------------

            logger.debug(
                "Running batch topic prediction. "
                "batch_size=%d",
                len(texts),
            )

            topics: list[TopicResult] = (
                self.topic_service.predict_batch(texts)
            )

            logger.info(
                "Batch topic prediction completed. "
                "results=%d",
                len(topics),
            )

            # -----------------------------------------------------------------
            # Validate Result Counts
            # -----------------------------------------------------------------

            expected_count = len(request.items)

            if len(languages) != expected_count:
                logger.error(
                    "Language model returned invalid result count. "
                    "expected=%d actual=%d",
                    expected_count,
                    len(languages),
                )

                raise ModelInferenceError(
                    "Language model returned an invalid number "
                    "of results."
                )

            if len(sentiments) != expected_count:
                logger.error(
                    "Sentiment model returned invalid result count. "
                    "expected=%d actual=%d",
                    expected_count,
                    len(sentiments),
                )

                raise ModelInferenceError(
                    "Sentiment model returned an invalid number "
                    "of results."
                )

            if len(topics) != expected_count:
                logger.error(
                    "Topic model returned invalid result count. "
                    "expected=%d actual=%d",
                    expected_count,
                    len(topics),
                )

                raise ModelInferenceError(
                    "Topic model returned an invalid number "
                    "of results."
                )

            # -----------------------------------------------------------------
            # Processing Time
            # -----------------------------------------------------------------

            processing_time_ms = self._processing_time_ms(
                started_at
            )

            # -----------------------------------------------------------------
            # Build Results
            # -----------------------------------------------------------------

            results: list[BatchAnalyzeResult] = []

            for item, language, sentiment, topic in zip(
                request.items,
                languages,
                sentiments,
                topics,
            ):
                results.append(
                    BatchAnalyzeResult(
                        id=item.id,
                        language=language,
                        sentiment=sentiment,
                        topic=topic,
                        processing_time_ms=processing_time_ms,
                    )
                )

            # -----------------------------------------------------------------
            # Response
            # -----------------------------------------------------------------

            response = BatchAnalyzeResponse(
                results=results,
                model=self._model_info(),
                processing_time_ms=processing_time_ms,
            )

            logger.info(
                "Batch analysis completed successfully. "
                "batch_size=%d processing_time_ms=%d",
                len(results),
                processing_time_ms,
            )

            return response

        except InvalidTextError:
            logger.warning(
                "Invalid text supplied in batch analysis."
            )
            raise

        except ModelInferenceError:
            logger.exception(
                "Model inference failed during batch analysis."
            )
            raise

        except Exception as exc:
            logger.exception(
                "Unexpected error during batch analysis. "
                "error_type=%s error=%s",
                type(exc).__name__,
                str(exc),
            )

            raise ModelInferenceError(
                "Failed to analyze batch."
            ) from exc

    # =========================================================================
    # Model Information
    # =========================================================================

    def _model_info(self) -> ModelInfo:
        """
        Return the currently configured model names and versions.
        """

        language_model_name = self._model_name(
            self.language_service,
            "LANGUAGE_MODEL_NAME",
            "municipal-language",
        )

        sentiment_model_name = self._model_name(
            self.sentiment_service,
            "SENTIMENT_MODEL_NAME",
            "municipal-sentiment",
        )

        topic_model_name = self._model_name(
            self.topic_service,
            "TOPIC_MODEL_NAME",
            "municipal-topic",
        )

        language_model_version = self._model_version(
            self.language_service,
            "LANGUAGE_MODEL_VERSION",
        )

        sentiment_model_version = self._model_version(
            self.sentiment_service,
            "SENTIMENT_MODEL_VERSION",
        )

        topic_model_version = self._model_version(
            self.topic_service,
            "TOPIC_MODEL_VERSION",
        )

        return ModelInfo(
            language=ModelVersion(
                name=language_model_name,
                version=language_model_version,
            ),
            sentiment=ModelVersion(
                name=sentiment_model_name,
                version=sentiment_model_version,
            ),
            topic=ModelVersion(
                name=topic_model_name,
                version=topic_model_version,
            ),
        )

    # =========================================================================
    # Model Configuration Helpers
    # =========================================================================

    @staticmethod
    def _model_version(
        service: object,
        setting_name: str,
    ) -> str:
        """
        Read a model version from the service settings.
        """

        settings = getattr(service, "settings", None)

        if settings is None:
            raise ModelInferenceError(
                "Model service settings are unavailable."
            )

        version = getattr(settings, setting_name, None)

        if not isinstance(version, str) or not version.strip():
            raise ModelInferenceError(
                f"Model setting '{setting_name}' is missing or invalid."
            )

        return version.strip()

    @staticmethod
    def _model_name(
        service: object,
        setting_name: str,
        default: str,
    ) -> str:
        """
        Read a model name from service settings when available.

        Falls back to the supplied stable application model name.
        """

        settings = getattr(service, "settings", None)

        if settings is None:
            return default

        value = getattr(settings, setting_name, None)

        if isinstance(value, str) and value.strip():
            return value.strip()

        return default

    # =========================================================================
    # Processing Time
    # =========================================================================

    @staticmethod
    def _processing_time_ms(
        started_at: float,
    ) -> int:
        """
        Calculate elapsed processing time in milliseconds.
        """

        elapsed = time.perf_counter() - started_at

        return max(
            0,
            int(round(elapsed * 1000)),
        )

    # =========================================================================
    # Validation
    # =========================================================================

    @staticmethod
    def _validate_text(text: str) -> str:
        """
        Validate and normalize text before inference.

        Pydantic already performs whitespace stripping on
        AnalyzeRequest, but this validation also protects
        direct service usage.
        """

        if not isinstance(text, str):
            raise InvalidTextError(
                "Text must be a string."
            )

        normalized = text.strip()

        if not normalized:
            raise InvalidTextError(
                "Text must not be empty."
            )

        return normalized