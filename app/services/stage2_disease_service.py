


"""
Stage 2: Disease Detection Service
"""
import numpy as np
import logging
from typing import Dict, Any, List

from app.services.model_loader import model_loader
from app.services.image_preprocessing import preprocessor
from app.config import settings

logger = logging.getLogger(__name__)


class DiseaseDetectionService:
    """Service for crop disease detection"""
    
    def __init__(self):
        self.models = {}
        self.disease_classes = {}
    
    def initialize(self):
        """Initialize service"""
        # Models are loaded via model_loader
        self.disease_classes = model_loader.disease_classes
        logger.info("Disease Detection Service initialized")
    
    async def predict_disease(
        self,
        image_bytes: bytes,
        crop_type: str
    ) -> Dict[str, Any]:
        """
        Predict disease for specific crop type
        
        Args:
            image_bytes: Raw image bytes
            crop_type: Type of crop (from Stage 1)
            
        Returns:
            Disease prediction results
        """
        try:
            # Get crop-specific model
            model = model_loader.get_stage2_model(crop_type)
            disease_classes = model_loader.get_disease_classes(crop_type)
            
            # Preprocess image
            processed_image = preprocessor.preprocess_for_stage2(image_bytes)
            batch_image = np.expand_dims(processed_image, axis=0)
            
            # Predict
            predictions = model.predict(batch_image, verbose=0)
            
            # Get top prediction
            disease_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][disease_idx])
            disease_name = disease_classes[disease_idx]
            
            # Determine severity based on confidence
            severity = self._calculate_severity(disease_name, confidence)
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {
                    'disease': disease_classes[idx],
                    'confidence': float(predictions[0][idx])
                }
                for idx in top_3_indices
            ]
            
            # Check if healthy
            is_healthy = disease_name.lower() == 'healthy'
            
            result = {
                'disease_name': disease_name,
                'confidence': confidence,
                'severity': severity,
                'is_healthy': is_healthy,
                'requires_treatment': not is_healthy and confidence > settings.DISEASE_CONFIDENCE_THRESHOLD,
                'top_3_predictions': top_3_predictions,
                'all_predictions': {
                    disease_classes[i]: float(predictions[0][i])
                    for i in range(len(disease_classes))
                }
            }
            
            logger.info(f"Disease prediction for {crop_type}: {disease_name} ({confidence:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in disease prediction: {str(e)}")
            raise
    
    def _calculate_severity(self, disease_name: str, confidence: float) -> str:
        """
        Calculate disease severity based on disease type and confidence
        
        Args:
            disease_name: Name of the disease
            confidence: Prediction confidence
            
        Returns:
            Severity level: 'healthy', 'low', 'moderate', 'high'
        """
        if disease_name.lower() == 'healthy':
            return 'healthy'
        
        # Severity based on confidence
        if confidence >= 0.85:
            return 'high'
        elif confidence >= 0.65:
            return 'moderate'
        else:
            return 'low'
    
    def get_disease_info(self, crop_type: str, disease_name: str) -> Dict[str, Any]:
        """
        Get basic disease information
        
        Args:
            crop_type: Type of crop
            disease_name: Name of disease
            
        Returns:
            Disease information
        """
        # Basic disease info (can be expanded)
        info = {
            'crop': crop_type,
            'disease': disease_name,
            'is_healthy': disease_name.lower() == 'healthy'
        }
        
        return info


# Global service instance
disease_service = DiseaseDetectionService()
