"""
Disease Detection API Endpoints
"""
import time
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional

from app.schemas.response import (
    DiseaseDetectionResponse,
    ErrorResponse,
    TreatmentRecommendations
)
from app.services.stage1_crop_service import crop_service
from app.services.stage2_disease_service import disease_service
from app.services.treatment_service import treatment_service
from app.services.image_preprocessing import preprocessor
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/detect-disease",
    response_model=DiseaseDetectionResponse,
    summary="Detect crop disease from leaf image",
    description="""
    Upload a leaf image to detect crop type and disease.
    
    Returns:
    - Crop identification
    - Disease detection
    - Treatment recommendations in Hindi and English
    - Severity assessment
    - Home remedies and chemical treatments
    
    For farmers: किसानों के लिए सरल और समझने योग्य उपचार जानकारी
    """
)
async def detect_disease(
    file: UploadFile = File(..., description="Leaf image (JPG, PNG)"),
    field_size: Optional[str] = Form("small", description="Farm size: 'small' or 'large'"),
    language: Optional[str] = Form("hi", description="Response language: 'hi' (Hindi) or 'en' (English)")
):
    """
    Main endpoint for disease detection
    
    Process:
    1. Validate image
    2. Stage 1: Identify crop type
    3. Stage 2: Detect disease
    4. Fetch treatment recommendations
    """
    start_time = time.time()
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Validate image
        is_valid, error_msg = preprocessor.validate_image(image_bytes, settings.MAX_IMAGE_SIZE)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Check if image contains a leaf
        contains_leaf, leaf_confidence = preprocessor.detect_leaf_in_image(image_bytes)
        if not contains_leaf:
            logger.warning(f"Image may not contain a leaf (confidence: {leaf_confidence:.2f})")
        
        # STAGE 1: Crop Type Classification
        logger.info("Starting Stage 1: Crop Classification")
        crop_result = await crop_service.predict_crop_type(image_bytes)
        
        if not crop_result['is_confident']:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Unable to identify crop type clearly",
                    "message_hindi": "फसल की पहचान स्पष्ट नहीं हो पा रही है। कृपया पत्ती की साफ तस्वीर अपलोड करें।",
                    "message_english": "Unable to identify crop type clearly. Please upload a clear leaf image.",
                    "suggestions": [
                        "Take photo in good lighting",
                        "Ensure leaf is in focus",
                        "Avoid blurry images",
                        "Show full leaf clearly"
                    ]
                }
            )
        
        crop_type = crop_result['crop_type']
        crop_confidence = crop_result['confidence']
        
        logger.info(f"Crop identified: {crop_type} ({crop_confidence:.2%})")
        
        # STAGE 2: Disease Detection
        logger.info(f"Starting Stage 2: Disease Detection for {crop_type}")
        disease_result = await disease_service.predict_disease(image_bytes, crop_type)
        
        disease_name = disease_result['disease_name']
        disease_confidence = disease_result['confidence']
        severity = disease_result['severity']
        is_healthy = disease_result['is_healthy']
        
        logger.info(f"Disease detected: {disease_name} ({disease_confidence:.2%}), Severity: {severity}")
        
        # Get treatment recommendations (only if not healthy)
        treatment_recommendations = None
        if not is_healthy and disease_result['requires_treatment']:
            logger.info("Fetching treatment recommendations")
            treatment_data = treatment_service.get_treatment_recommendations(
                crop_type=crop_type,
                disease_name=disease_name,
                severity=severity,
                field_size=field_size
            )
            treatment_recommendations = treatment_data
        
        # Get crop info
        crop_info = crop_service.get_crop_info(crop_type)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Build response
        response = DiseaseDetectionResponse(
            success=True,
            message="रोग पहचान सफलतापूर्वक पूर्ण हुई" if language == "hi" else "Disease detection completed successfully",
            crop_type=crop_type,
            crop_confidence=crop_confidence,
            crop_name_hindi=crop_info.get('name_hindi', crop_type),
            disease_name=disease_name,
            disease_confidence=disease_confidence,
            severity=severity,
            is_healthy=is_healthy,
            treatment_recommendations=treatment_recommendations,
            processing_time_ms=processing_time,
            model_version="1.0.0"
        )
        
        logger.info(f"Detection completed in {processing_time:.2f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in disease detection: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "message_hindi": "सर्वर में त्रुटि। कृपया पुनः प्रयास करें।"
            }
        )


@router.post(
    "/identify-crop",
    summary="Identify crop type only (Stage 1)",
    description="Upload leaf image to identify crop type without disease detection"
)
async def identify_crop_only(
    file: UploadFile = File(..., description="Leaf image")
):
    """
    Endpoint for crop identification only (Stage 1)
    Useful for testing or when only crop type is needed
    """
    try:
        image_bytes = await file.read()
        
        # Validate
        is_valid, error_msg = preprocessor.validate_image(image_bytes)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Predict crop
        crop_result = await crop_service.predict_crop_type(image_bytes)
        
        return {
            "success": True,
            "crop_type": crop_result['crop_type'],
            "confidence": crop_result['confidence'],
            "is_confident": crop_result['is_confident'],
            "top_3_predictions": crop_result['top_3_predictions'],
            "crop_info": crop_service.get_crop_info(crop_result['crop_type'])
        }
        
    except Exception as e:
        logger.error(f"Error in crop identification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/get-treatment",
    summary="Get treatment recommendations",
    description="Get treatment recommendations for a specific crop disease"
)
async def get_treatment_info(
    crop_type: str = Form(..., description="Crop type (e.g., 'maize', 'wheat')"),
    disease_name: str = Form(..., description="Disease name (e.g., 'blight', 'rust')"),
    severity: str = Form("moderate", description="Severity: 'low', 'moderate', or 'high'"),
    field_size: str = Form("small", description="Field size: 'small' or 'large'")
):
    """
    Get treatment recommendations without uploading an image
    Useful when disease is already known
    """
    try:
        treatment_data = treatment_service.get_treatment_recommendations(
            crop_type=crop_type,
            disease_name=disease_name,
            severity=severity,
            field_size=field_size
        )
        
        return {
            "success": True,
            "crop_type": crop_type,
            "disease_name": disease_name,
            "severity": severity,
            "field_size": field_size,
            "treatment_recommendations": treatment_data
        }
        
    except Exception as e:
        logger.error(f"Error getting treatment info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/prevention-tips/{crop_type}",
    summary="Get general prevention tips",
    description="Get general disease prevention tips for a crop"
)
async def get_prevention_tips(crop_type: str):
    """
    Get general prevention tips for a specific crop
    """
    try:
        tips = treatment_service.get_general_prevention_tips(crop_type)
        
        return {
            "success": True,
            "crop_type": crop_type,
            "prevention_tips": tips
        }
        
    except Exception as e:
        logger.error(f"Error getting prevention tips: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/supported-crops",
    summary="Get list of supported crops",
    description="Returns all crops supported by the system"
)
async def get_supported_crops():
    """
    Get list of all supported crops
    """
    crops = crop_service.get_crop_classes()
    
    crop_details = []
    for crop in crops:
        info = crop_service.get_crop_info(crop)
        crop_details.append({
            "crop_name": crop,
            "hindi_name": info.get('name_hindi', ''),
            "scientific_name": info.get('scientific_name', ''),
            "common_diseases": info.get('common_diseases', [])
        })
    
    return {
        "success": True,
        "total_crops": len(crops),
        "crops": crop_details
    }


@router.get(
    "/diseases/{crop_type}",
    summary="Get diseases for a crop",
    description="Returns all diseases that can be detected for a specific crop"
)
async def get_crop_diseases(crop_type: str):
    """
    Get list of diseases for a specific crop
    """
    try:
        disease_classes = disease_service.disease_classes.get(crop_type, [])
        
        if not disease_classes:
            raise HTTPException(
                status_code=404,
                detail=f"Crop '{crop_type}' not found or not supported"
            )
        
        return {
            "success": True,
            "crop_type": crop_type,
            "total_diseases": len(disease_classes),
            "diseases": disease_classes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting crop diseases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
