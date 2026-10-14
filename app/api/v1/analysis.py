from fastapi import APIRouter, Depends, status

from app.api.v1.security_test import verify_internal_api_key
from app.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
)
from app.services.analysis_service import AnalysisService


router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"],
)


def get_analysis_service() -> AnalysisService:
    """
    Dependency provider for the shared analysis service.

    The AnalysisService is implemented as a singleton so that
    expensive AI models are loaded once and reused across requests.
    """
    return AnalysisService.get_instance()


# =============================================================================
# Single Analysis
# =============================================================================

@router.post(
    "",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a single text",
    description=(
        "Detect the language and predict the sentiment of a single text."
    ),
    dependencies=[Depends(verify_internal_api_key)],
)
def analyze(
    request: AnalyzeRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalyzeResponse:
    """
    Analyze one text through the municipal AI pipeline.

    Pipeline:

        Request
          ↓
        Internal API Key Authentication
          ↓
        AnalysisService
          ↓
        LanguageService
          ↓
        SentimentService
          ↓
        AnalyzeResponse
    """
    return service.analyze(request)


# =============================================================================
# Batch Analysis
# =============================================================================

@router.post(
    "/batch",
    response_model=BatchAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze multiple texts",
    description=(
        "Detect language and predict sentiment for multiple texts "
        "using batch inference."
    ),
    dependencies=[Depends(verify_internal_api_key)],
)
def analyze_batch(
    request: BatchAnalyzeRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> BatchAnalyzeResponse:
    """
    Analyze multiple texts through the municipal AI pipeline.

    Batch inference allows transformer models to process multiple
    texts together and is preferred for larger workloads.

    Client-provided IDs are preserved in the response.
    """
    return service.analyze_batch(request)