from __future__ import annotations

import json
import logging
from typing import Any

from app.core.exceptions import (
    InvalidTextError,
    ModelInferenceError,
)
from app.services.providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class InsightService:
    """
    Application-level service for municipal AI insights.

    Responsibilities:
        - Prepare structured municipal evidence for the LLM.
        - Build a controlled insight prompt.
        - Call the configured LLM provider.
        - Parse the generated response.
        - Validate the generated insight structure.

    This service does not communicate with Ollama directly.
    Ollama communication is handled by OllamaProvider.
    """

    _instance: "InsightService | None" = None

    def __init__(
        self,
        ollama_provider: OllamaProvider | None = None,
    ) -> None:
        logger.info("Initializing InsightService.")

        try:
            self.ollama_provider = (
                ollama_provider
                or OllamaProvider.get_instance()
            )

            logger.info(
                "InsightService initialized successfully."
            )

        except Exception:
            logger.exception(
                "Failed to initialize InsightService."
            )
            raise

    @classmethod
    def get_instance(cls) -> "InsightService":
        """
        Return the shared InsightService instance.
        """

        if cls._instance is None:
            logger.info(
                "Creating InsightService singleton instance."
            )
            cls._instance = cls()

        else:
            logger.debug(
                "Reusing existing InsightService singleton."
            )

        return cls._instance

    def generate_post_insight(
        self,
        *,
        post_text: str,
        comments: list[dict[str, Any]] | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a municipal insight for a Facebook post.

        Args:
            post_text:
                Original Facebook post text.

            comments:
                Optional list of analyzed comments.

                Example:

                    [
                        {
                            "text": "...",
                            "language": "om",
                            "sentiment": "negative",
                            "topic": "roads_infrastructure",
                        }
                    ]

            analysis:
                Optional aggregate AI analysis.

                Example:

                    {
                        "language": {
                            "code": "om",
                            "confidence": 0.99,
                        },
                        "sentiment": {
                            "negative": 62,
                            "neutral": 21,
                            "positive": 17,
                        },
                        "topics": {
                            "roads_infrastructure": 54,
                            "water": 12,
                        },
                    }

        Returns:
            Structured municipal insight dictionary.

        Raises:
            InvalidTextError:
                If the post text is empty.

            ModelInferenceError:
                If insight generation or response parsing fails.
        """

        normalized_post = self._validate_text(post_text)

        normalized_comments = self._normalize_comments(
            comments or []
        )

        normalized_analysis = self._normalize_analysis(
            analysis or {}
        )

        logger.info(
            "Starting post insight generation. "
            "post_length=%d comments=%d",
            len(normalized_post),
            len(normalized_comments),
        )

        prompt = self._build_post_prompt(
            post_text=normalized_post,
            comments=normalized_comments,
            analysis=normalized_analysis,
        )

        try:
            generated_text = self.ollama_provider.generate(
                prompt,
                temperature=0.2,
                max_tokens=1000,
            )

            logger.debug(
                "Raw insight response received. "
                "response_length=%d",
                len(generated_text),
            )

            insight = self._parse_response(
                generated_text
            )

            self._validate_insight(insight)

            logger.info(
                "Post insight generated successfully."
            )

            return insight

        except InvalidTextError:
            raise

        except ModelInferenceError:
            logger.exception(
                "LLM inference failed during post insight generation."
            )
            raise

        except Exception as exc:
            logger.exception(
                "Unexpected error during post insight generation. "
                "error_type=%s error=%s",
                type(exc).__name__,
                str(exc),
            )

            raise ModelInferenceError(
                "Failed to generate municipal post insight."
            ) from exc

    def _build_post_prompt(
        self,
        *,
        post_text: str,
        comments: list[dict[str, Any]],
        analysis: dict[str, Any],
    ) -> str:
        """
        Build a controlled prompt for municipal insight generation.

        The LLM is explicitly instructed to interpret only the
        supplied evidence and not invent facts.
        """

        comments_json = json.dumps(
            comments,
            ensure_ascii=False,
            indent=2,
        )

        analysis_json = json.dumps(
            analysis,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
You are a municipal public-service intelligence assistant.

Your task is to analyze a Facebook post and its available
AI-classification results and produce a concise executive
insight for municipal decision-makers.

IMPORTANT RULES:

1. Use ONLY the information provided in the input.
2. Do NOT invent facts, events, numbers, locations, people,
   causes, actions, or outcomes.
3. Do NOT claim that something happened unless the evidence
   supports it.
4. Distinguish clearly between:
   - reported problems
   - public appreciation
   - public requests
   - recommendations or expectations
5. Do not treat every negative statement as a confirmed
   municipal failure.
6. Do not infer motives or intentions.
7. Preserve uncertainty when the evidence is ambiguous.
8. The output MUST be valid JSON.
9. Do not use Markdown.
10. Do not include explanations outside the JSON object.
11. Keep the summary concise and suitable for a municipal
    executive dashboard.
12. The original comments may contain Afaan Oromo, Amharic,
    or English. Understand them in context and write the
    final insight in English.

POST:

{post_text}

AGGREGATED AI ANALYSIS:

{analysis_json}

ANALYZED COMMENTS:

{comments_json}

Return exactly this JSON structure:

{{
  "summary": "A concise executive summary of the main public reaction.",
  "key_concerns": [
    "Concern supported by the supplied evidence."
  ],
  "positive_signals": [
    "Positive signal supported by the supplied evidence."
  ],
  "public_requests": [
    "Request or expectation expressed by the public."
  ],
  "attention_areas": [
    "Area that municipal staff may need to monitor based on the evidence."
  ]
}}

Additional requirements:

- Use empty arrays when there is no evidence for a category.
- Do not duplicate the same statement across multiple arrays.
- Do not create recommendations that are not supported by
  the comments or analysis.
- Do not mention sentiment percentages unless they are
  provided in the input and materially useful.
- The "attention_areas" field should describe an evidence-based
  area to monitor, not issue an unsupported instruction.
""".strip()

    @staticmethod
    def _validate_text(text: str) -> str:
        """
        Validate and normalize text.
        """

        if not isinstance(text, str):
            raise InvalidTextError(
                "Post text must be a string."
            )

        normalized = text.strip()

        if not normalized:
            raise InvalidTextError(
                "Post text must not be empty."
            )

        return normalized

    @staticmethod
    def _normalize_comments(
        comments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Normalize comment data before sending it to the LLM.

        Only fields useful for municipal insight generation
        are retained.
        """

        normalized: list[dict[str, Any]] = []

        for index, comment in enumerate(comments):
            if not isinstance(comment, dict):
                logger.warning(
                    "Skipping invalid comment at index=%d.",
                    index,
                )
                continue

            text = comment.get("text")

            if not isinstance(text, str):
                logger.warning(
                    "Skipping comment without valid text. index=%d",
                    index,
                )
                continue

            text = text.strip()

            if not text:
                continue

            item: dict[str, Any] = {
                "text": text,
            }

            for field in (
                "language",
                "sentiment",
                "sentiment_confidence",
                "topic",
                "topic_confidence",
            ):
                if field in comment:
                    item[field] = comment[field]

            normalized.append(item)

        return normalized

    @staticmethod
    def _normalize_analysis(
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize aggregate analysis.

        This method intentionally does not calculate statistics.
        Aggregation should happen in the classification/application
        layer where the underlying records are available.
        """

        if not isinstance(analysis, dict):
            logger.warning(
                "Invalid aggregate analysis supplied. "
                "Using empty analysis."
            )
            return {}

        return analysis

    @staticmethod
    def _parse_response(
        generated_text: str,
    ) -> dict[str, Any]:
        """
        Parse JSON generated by the LLM.

        Handles the common case where a model accidentally wraps
        JSON in a Markdown code fence.
        """

        if not isinstance(generated_text, str):
            raise ModelInferenceError(
                "LLM returned an invalid response type."
            )

        cleaned = generated_text.strip()

        if not cleaned:
            raise ModelInferenceError(
                "LLM returned an empty insight."
            )

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()

        try:
            data = json.loads(cleaned)

        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse LLM insight JSON. "
                "response=%s",
                cleaned[:2000],
            )

            raise ModelInferenceError(
                "LLM returned invalid insight JSON."
            ) from exc

        if not isinstance(data, dict):
            raise ModelInferenceError(
                "LLM insight response must be a JSON object."
            )

        return data

    @staticmethod
    def _validate_insight(
        insight: dict[str, Any],
    ) -> None:
        """
        Validate the required insight structure.

        This protects the API from returning malformed
        LLM-generated content.
        """

        required_fields = (
            "summary",
            "key_concerns",
            "positive_signals",
            "public_requests",
            "attention_areas",
        )

        for field in required_fields:
            if field not in insight:
                raise ModelInferenceError(
                    f"LLM insight is missing required field '{field}'."
                )

        if not isinstance(
            insight["summary"],
            str,
        ):
            raise ModelInferenceError(
                "LLM insight 'summary' must be a string."
            )

        if not insight["summary"].strip():
            raise ModelInferenceError(
                "LLM insight 'summary' must not be empty."
            )

        list_fields = (
            "key_concerns",
            "positive_signals",
            "public_requests",
            "attention_areas",
        )

        for field in list_fields:
            value = insight[field]

            if not isinstance(value, list):
                raise ModelInferenceError(
                    f"LLM insight '{field}' must be an array."
                )

            for item in value:
                if not isinstance(item, str):
                    raise ModelInferenceError(
                        f"LLM insight '{field}' "
                        "must contain only strings."
                    )

        logger.debug(
            "LLM insight structure validated successfully."
        )