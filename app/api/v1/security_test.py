from fastapi import APIRouter, Depends

from app.core.security import verify_internal_api_key


router = APIRouter(
    prefix="/security-test",
    tags=["Security"],
)


@router.get(
    "",
    dependencies=[Depends(verify_internal_api_key)],
)
async def security_test() -> dict[str, str]:
    return {
        "status": "authenticated",
        "service": "municipal-ai-service",
    }