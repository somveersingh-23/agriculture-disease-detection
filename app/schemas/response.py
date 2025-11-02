"""
Pydantic schemas for API responses
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class HomeRemedy(BaseModel):
    """Home remedy details"""
    treatment: str
    ingredients: str
    preparation: str
    application: str
    cost_estimate: str


class ChemicalTreatment(BaseModel):
    """Chemical treatment details"""
    product: str
    dosage: str
    method: str
    frequency: Optional[str] = None
    cost: str


class TreatmentRecommendations(BaseModel):
    """Complete treatment recommendations"""
    disease_name_hindi: str
    disease_name_english: str
    symptoms: List[str]
    home_remedies: List[Dict[str, str]]
    chemical_treatment: Dict[str, Any]
    prevention: List[str]
    severity_action: str
    expert_contact: str
    recommended_approach: str
    urgent_action_required: bool


class TopPrediction(BaseModel):
    """Top prediction details"""
    crop: Optional[str] = None
    disease: Optional[str] = None
    confidence: float


class CropPredictionResponse(BaseModel):
    """Stage 1: Crop prediction response"""
    crop_type: str
    confidence: float
    is_confident: bool
    top_3_predictions: List[TopPrediction]


class DiseasePredictionResponse(BaseModel):
    """Stage 2: Disease prediction response"""
    disease_name: str
    confidence: float
    severity: str
    is_healthy: bool
    requires_treatment: bool
    top_3_predictions: List[TopPrediction]


class DiseaseDetectionResponse(BaseModel):
    """Complete disease detection response"""
    success: bool
    message: str
    
    # Stage 1 results
    crop_type: str
    crop_confidence: float
    crop_name_hindi: Optional[str] = None
    
    # Stage 2 results
    disease_name: str
    disease_confidence: float
    severity: str
    is_healthy: bool
    
    # Treatment information
    treatment_recommendations: Optional[TreatmentRecommendations] = None
    
    # Additional info
    processing_time_ms: Optional[float] = None
    model_version: str = "1.0.0"
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Disease detection completed successfully",
                "crop_type": "maize",
                "crop_confidence": 0.95,
                "crop_name_hindi": "मक्का",
                "disease_name": "blight",
                "disease_confidence": 0.88,
                "severity": "moderate",
                "is_healthy": False,
                "treatment_recommendations": {
                    "disease_name_hindi": "मक्का झुलसा रोग",
                    "disease_name_english": "Maize Blight",
                    "symptoms": ["पत्तों पर लंबे भूरे धब्बे"],
                    "home_remedies": [],
                    "chemical_treatment": {},
                    "prevention": [],
                    "severity_action": "घरेलू उपाय से शुरुआत करें",
                    "expert_contact": "1800-180-1551",
                    "recommended_approach": "पहले घरेलू उपाय आजमाएं",
                    "urgent_action_required": False
                },
                "processing_time_ms": 1250.5,
                "model_version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    version: str
    supported_crops: List[str]
