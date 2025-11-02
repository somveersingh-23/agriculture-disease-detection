"""
Health check and system status endpoints
"""
from fastapi import APIRouter
from app.schemas.response import HealthCheckResponse
from app.services.model_loader import model_loader
from app.services.stage1_crop_service import crop_service
from app.config import settings

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check if API is running and models are loaded"
)
async def health_check():
    """
    Health check endpoint
    Returns system status and model information
    """
    return HealthCheckResponse(
        status="healthy",
        models_loaded=model_loader._models_loaded,
        version=settings.APP_VERSION,
        supported_crops=crop_service.get_crop_classes()
    )


@router.get(
    "/",
    summary="Root endpoint",
    description="Welcome message and API information"
)
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "कृषि रोग पहचान API - Agriculture Disease Detection API",
        "version": settings.APP_VERSION,
        "description": "किसानों के लिए फसल रोग पहचान और उपचार सुझाव",
        "endpoints": {
            "detect_disease": "/api/v1/detect-disease",
            "identify_crop": "/api/v1/identify-crop",
            "get_treatment": "/api/v1/get-treatment",
            "supported_crops": "/api/v1/supported-crops",
            "health": "/health"
        },
        "documentation": "/docs",
        "supported_languages": ["Hindi (हिंदी)", "English"],
        "contact": "किसान कॉल सेंटर: 1800-180-1551"
    }


@router.get(
    "/status",
    summary="System status",
    description="Detailed system status and statistics"
)
async def system_status():
    """
    Detailed system status
    """
    return {
        "api_status": "running",
        "models": {
            "stage1_loaded": model_loader._models_loaded,
            "stage2_models_count": len(model_loader.stage2_models),
            "total_crops": len(crop_service.get_crop_classes())
        },
        "configuration": {
            "max_image_size_mb": settings.MAX_IMAGE_SIZE / (1024 * 1024),
            "supported_image_types": settings.ALLOWED_IMAGE_TYPES,
            "crop_confidence_threshold": settings.CROP_CONFIDENCE_THRESHOLD,
            "disease_confidence_threshold": settings.DISEASE_CONFIDENCE_THRESHOLD
        },
        "version": settings.APP_VERSION
    }
