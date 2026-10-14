from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1.analysis import router as analysis_router
from app.api.v1.health import router as health_router
from app.api.v1.insight import router as insight_router
from app.api.v1.security_test import router as security_test_router

from app.core.config import get_settings
from app.core.exceptions import (
    AIServiceError,
    InvalidTextError,
    ModelInferenceError,
    ModelLoadError,
)

from app.core.logging import (
    configure_logging,
    get_logger,
)

from app.services.language_service import LanguageService
from app.services.sentiment_service import SentimentService
from app.services.topic_service import TopicService


# =============================================================================
# Configuration
# =============================================================================

settings = get_settings()

configure_logging(
    settings.LOG_LEVEL,
)

logger = get_logger(__name__)


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Application startup and shutdown lifecycle.

    AI models are loaded and validated during startup so that
    model configuration/load failures are detected before the
    service begins accepting inference traffic.
    """

    logger.info(
        "Starting Municipal AI Service. "
        "environment=%s version=%s device=%s",
        settings.APP_ENV,
        settings.APP_VERSION,
        settings.MODEL_DEVICE,
    )

    try:
        # ---------------------------------------------------------------------
        # Load Language Model
        # ---------------------------------------------------------------------

        LanguageService.get_instance()

        logger.info(
            "Language model loaded successfully. "
            "version=%s",
            settings.LANGUAGE_MODEL_VERSION,
        )

        # ---------------------------------------------------------------------
        # Load Sentiment Model
        # ---------------------------------------------------------------------

        SentimentService.get_instance()

        logger.info(
            "Sentiment model loaded successfully. "
            "version=%s labels=%s",
            settings.SENTIMENT_MODEL_VERSION,
            settings.SENTIMENT_NUM_LABELS,
        )

        # ---------------------------------------------------------------------
        # Load Topic Model
        # ---------------------------------------------------------------------

        TopicService.get_instance()

        logger.info(
            "Topic model loaded successfully. "
            "version=%s topics=%s",
            settings.TOPIC_MODEL_VERSION,
            settings.TOPIC_NUM_LABELS,
        )

    except ModelLoadError:
        logger.exception(
            "Failed to load one or more AI models during startup."
        )
        raise

    logger.info(
        "Municipal AI Service started successfully."
    )

    yield

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    logger.info(
        "Shutting down Municipal AI Service."
    )


# =============================================================================
# Application Factory
# =============================================================================

def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """

    application = FastAPI(
        title=settings.APP_NAME,
        description=(
            "AI inference service for the "
            "Municipal Intelligence Platform."
        ),
        version=settings.APP_VERSION,
        debug=settings.APP_DEBUG,
        lifespan=lifespan,
    )

    logger.info(
        "Creating FastAPI application. "
        "app=%s version=%s environment=%s",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.APP_ENV,
    )

    # =========================================================================
    # Exception Handlers
    # =========================================================================

    @application.exception_handler(InvalidTextError)
    async def invalid_text_handler(
        request: Request,
        exc: InvalidTextError,
    ) -> JSONResponse:
        """
        Handle invalid text errors.
        """

        logger.warning(
            "Invalid text request. "
            "method=%s path=%s error=%s",
            request.method,
            request.url.path,
            str(exc),
        )

        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "INVALID_TEXT",
                    "message": str(exc),
                }
            },
        )

    @application.exception_handler(ModelInferenceError)
    async def model_inference_handler(
        request: Request,
        exc: ModelInferenceError,
    ) -> JSONResponse:
        """
        Handle AI model inference errors.

        Detailed exceptions are logged internally while the
        client receives a safe generic error message.
        """

        logger.exception(
            "Model inference error. "
            "method=%s path=%s error=%s",
            request.method,
            request.url.path,
            str(exc),
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "MODEL_INFERENCE_ERROR",
                    "message": "AI inference failed.",
                }
            },
        )

    @application.exception_handler(AIServiceError)
    async def ai_service_error_handler(
        request: Request,
        exc: AIServiceError,
    ) -> JSONResponse:
        """
        Handle general AI service errors.
        """

        logger.exception(
            "AI service error. "
            "method=%s path=%s error=%s",
            request.method,
            request.url.path,
            str(exc),
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "AI_SERVICE_ERROR",
                    "message": str(exc),
                }
            },
        )

    # =========================================================================
    # API Routes
    # =========================================================================

    application.include_router(
        health_router,
        prefix=settings.API_PREFIX,
    )

    application.include_router(
        security_test_router,
        prefix=settings.API_PREFIX,
    )

    application.include_router(
        analysis_router,
        prefix=settings.API_PREFIX,
    )

    application.include_router(
        insight_router,
        prefix=settings.API_PREFIX,
    )

    logger.info(
        "API routes registered successfully."
    )

    return application


# =============================================================================
# Application Instance
# =============================================================================

app = create_application()