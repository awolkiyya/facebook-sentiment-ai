from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Insight Evidence
# =============================================================================

class InsightComment(BaseModel):
    """
    An analyzed Facebook comment supplied as evidence
    for municipal insight generation.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=5_000,
        description="Original Facebook comment text.",
    )

    language: str | None = Field(
        default=None,
        description="Detected language code.",
        examples=["om", "am", "en"],
    )

    sentiment: str | None = Field(
        default=None,
        description="Detected sentiment.",
        examples=["negative", "neutral", "positive"],
    )

    sentiment_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sentiment prediction confidence.",
    )

    topic: str | None = Field(
        default=None,
        description="Detected municipal topic.",
        examples=["roads_infrastructure"],
    )

    topic_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Topic prediction confidence.",
    )


# =============================================================================
# Aggregate Sentiment
# =============================================================================

class InsightSentimentDistribution(BaseModel):
    """
    Aggregate sentiment distribution for the analyzed comments.

    Values are represented as percentages from 0 to 100.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    negative: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of negative comments.",
    )

    neutral: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of neutral comments.",
    )

    positive: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of positive comments.",
    )


# =============================================================================
# Aggregate Analysis
# =============================================================================

class InsightAnalysis(BaseModel):
    """
    Structured AI analysis supplied to the insight engine.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    language: str | None = Field(
        default=None,
        description="Primary language associated with the post.",
        examples=["om", "am", "en"],
    )

    language_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Language detection confidence.",
    )

    sentiment: InsightSentimentDistribution | None = Field(
        default=None,
        description="Aggregate sentiment distribution.",
    )

    topics: dict[str, int] = Field(
        default_factory=dict,
        description="Topic counts across analyzed comments.",
    )


# =============================================================================
# Insight Request
# =============================================================================

class InsightRequest(BaseModel):
    """
    Request for generating a municipal insight from a Facebook post.

    The request can contain:
        - the original post
        - analyzed comments
        - aggregate AI analysis
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    post_text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Original Facebook post text.",
    )

    comments: list[InsightComment] = Field(
        default_factory=list,
        max_length=500,
        description="Analyzed Facebook comments used as evidence.",
    )

    analysis: InsightAnalysis | None = Field(
        default=None,
        description="Aggregate AI analysis of the post and comments.",
    )


# =============================================================================
# Insight Content
# =============================================================================

class InsightContent(BaseModel):
    """
    Generated municipal insight content.

    All fields are evidence-based interpretations generated
    by the local LLM.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    summary: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="Concise executive summary.",
    )

    key_concerns: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Key concerns supported by the evidence.",
    )

    positive_signals: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Positive signals supported by the evidence.",
    )

    public_requests: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Requests or expectations expressed by the public.",
    )

    attention_areas: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Evidence-based areas that may require monitoring.",
    )


# =============================================================================
# Insight Response
# =============================================================================

class InsightResponse(BaseModel):
    """
    Complete municipal post insight response.
    """

    post_text: str = Field(
        ...,
        description="Original Facebook post text.",
    )

    insight: InsightContent = Field(
        ...,
        description="Generated municipal insight.",
    )

    model: str = Field(
        ...,
        min_length=1,
        description="LLM model used to generate the insight.",
        examples=["llama3.2:3b"],
    )

    processing_time_ms: int | None = Field(
        default=None,
        ge=0,
        description="Insight generation processing time in milliseconds.",
    )