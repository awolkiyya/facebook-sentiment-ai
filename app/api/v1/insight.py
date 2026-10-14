from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, status

from app.core.security import verify_internal_api_key
from app.schemas.insight import (
    InsightRequest,
    InsightResponse,
)
from app.services.insight_service import InsightService

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/insight",
    tags=["Insight"],
)


@router.post(
    "",
    response_model=InsightResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate municipal insight",
    description=(
        "Generate an executive municipal insight from a Facebook "
        "post, analyzed comments, and structured AI analysis."
    ),
    dependencies=[
        Depends(verify_internal_api_key),
    ],
)
async def generate_insight(
    request: InsightRequest,
) -> InsightResponse:
    """
    Generate a municipal insight for a Facebook post.

    The endpoint accepts:
        - Original Facebook post text
        - Optional analyzed comments
        - Optional aggregate AI analysis

    The InsightService handles:
        - Prompt construction
        - Local LLM inference
        - Response parsing
        - Insight validation
    """

    logger.info(
        "Received insight generation request. "
        "post_length=%d comments=%d",
        len(request.post_text),
        len(request.comments),
    )

    started_at = time.perf_counter()

    try:
        service = InsightService.get_instance()

        comments = [
            comment.model_dump(
                exclude_none=True,
            )
            for comment in request.comments
        ]

        analysis = (
            request.analysis.model_dump(
                exclude_none=True,
            )
            if request.analysis is not None
            else {}
        )

        insight = service.generate_post_insight(
            post_text=request.post_text,
            comments=comments,
            analysis=analysis,
        )

        processing_time_ms = max(
            0,
            int(
                round(
                    (time.perf_counter() - started_at)
                    * 1000
                )
            ),
        )

        response = InsightResponse(
            post_text=request.post_text,
            insight=insight,
            model=service.ollama_provider.model,
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            "Insight generation completed successfully. "
            "processing_time_ms=%d",
            processing_time_ms,
        )

        return response

    except Exception:
        logger.exception(
            "Insight generation request failed."
        )
        raise