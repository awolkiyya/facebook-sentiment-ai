from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Single Analysis Request
# =============================================================================

class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=5_000,
        description="Text to analyze.",
    )


# =============================================================================
# Language
# =============================================================================

class LanguageResult(BaseModel):
    """
    Language detection result.

    Supported languages:

        om = Afaan Oromo
        am = Amharic
        en = English
    """

    code: str = Field(
        ...,
        description="Detected language code.",
        examples=["om", "am", "en"],
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model predicted probability.",
    )


# =============================================================================
# Sentiment
# =============================================================================

class SentimentResult(BaseModel):
    """
    Sentiment classification result.

    Labels:

        0 = negative
        1 = neutral
        2 = positive
    """

    label: str = Field(
        ...,
        description="Predicted sentiment label.",
        examples=["negative", "neutral", "positive"],
    )

    label_id: int = Field(
        ...,
        ge=0,
        le=2,
        description="Numeric sentiment class ID.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model predicted probability.",
    )

    positive_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    neutral_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    negative_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


# =============================================================================
# Topic
# =============================================================================

class TopicResult(BaseModel):
    """
    Municipal topic classification result.

    The topic is one of the 25 topics defined by the
    final municipal topic taxonomy.
    """

    topic: str = Field(
        ...,
        min_length=1,
        description="Predicted municipal topic.",
        examples=["roads_infrastructure"],
    )

    topic_id: int = Field(
        ...,
        ge=0,
        le=24,
        description="Numeric topic class ID.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model predicted probability.",
    )


# =============================================================================
# Model Information
# =============================================================================

class ModelVersion(BaseModel):
    """
    Model name and deployed version.
    """

    name: str = Field(
        ...,
        min_length=1,
    )

    version: str = Field(
        ...,
        min_length=1,
    )


class ModelInfo(BaseModel):
    """
    Models used to produce the analysis result.
    """

    language: ModelVersion

    sentiment: ModelVersion

    topic: ModelVersion


# =============================================================================
# Single Analysis Response
# =============================================================================

class AnalyzeResponse(BaseModel):
    """
    Complete municipal AI analysis result.
    """

    text: str

    language: LanguageResult

    sentiment: SentimentResult

    topic: TopicResult

    model: ModelInfo

    processing_time_ms: int | None = Field(
        default=None,
        ge=0,
    )


# =============================================================================
# Batch Analysis
# =============================================================================

class BatchAnalyzeItem(BaseModel):
    """
    Single item inside a batch request.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    id: str = Field(
        ...,
        min_length=1,
        max_length=255,
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=5_000,
    )


class BatchAnalyzeRequest(BaseModel):
    """
    Batch analysis request.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    items: list[BatchAnalyzeItem] = Field(
        ...,
        min_length=1,
        max_length=500,
    )


class BatchAnalyzeResult(BaseModel):
    """
    Analysis result for one batch item.
    """

    id: str

    language: LanguageResult

    sentiment: SentimentResult

    topic: TopicResult

    processing_time_ms: int | None = Field(
        default=None,
        ge=0,
    )


class BatchAnalyzeResponse(BaseModel):
    """
    Complete batch analysis response.
    """

    results: list[BatchAnalyzeResult]

    model: ModelInfo

    processing_time_ms: int | None = Field(
        default=None,
        ge=0,
    )
