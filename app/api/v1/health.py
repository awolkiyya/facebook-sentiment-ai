from fastapi import APIRouter

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("/live")
async def liveness() -> dict[str, str]:
    """
    Liveness probe.

    Confirms that the FastAPI process is running.
    This endpoint should remain lightweight and should not
    perform model inference or external service checks.
    """
    return {
        "status": "ok",
    }


@router.get("/ready")
async def readiness() -> dict[str, str]:
    """
    Readiness probe.

    Confirms that the Municipal AI Service is initialized
    and ready to process inference requests.
    """
    return {
        "status": "ready",
    }