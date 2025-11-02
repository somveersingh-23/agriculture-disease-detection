"""
Stage 1: Crop Type Classification Service
"""
import numpy as np
import logging
from typing import Tuple, Dict, Any

from app.services.model_loader import model_loader
from app.services.image_preprocessing import preprocessor
from app.config import settings

logger = logging.getLogger(__name__)


class CropClassificationService:
    """Service for crop type classification"""
    
    def __init__(self):
        self.model = None
        self.crop_classes = []
    
    def initialize(self):
        """Initialize service with loaded models"""
        self.model = model_loader.get_stage1_model()
        self.crop_classes = model_loader.get_crop_classes()
        logger.info("Crop Classification Service initialized")
    
    async def predict_crop_type(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict crop type from leaf image
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with crop prediction results
        """
        try:
            # Preprocess image
            processed_image = preprocessor.preprocess_for_stage1(image_bytes)
            
            # Add batch dimension
            batch_image = np.expand_dims(processed_image, axis=0)
            
            # Predict
            predictions = self.model.predict(batch_image, verbose=0)
            
            # Get top prediction
            crop_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][crop_idx])
            crop_type = self.crop_classes[crop_idx]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {
                    'crop': self.crop_classes[idx],
                    'confidence': float(predictions[0][idx])
                }
                for idx in top_3_indices
            ]
            
            # Check confidence threshold
            is_confident = confidence >= settings.CROP_CONFIDENCE_THRESHOLD
            
            result = {
                'crop_type': crop_type,
                'confidence': confidence,
                'is_confident': is_confident,
                'top_3_predictions': top_3_predictions,
                'all_predictions': {
                    self.crop_classes[i]: float(predictions[0][i])
                    for i in range(len(self.crop_classes))
                }
            }
            
            logger.info(f"Crop prediction: {crop_type} ({confidence:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in crop prediction: {str(e)}")
            raise
    
    def get_crop_info(self, crop_name: str) -> Dict[str, Any]:
        """
        Get information about a crop
        
        Args:
            crop_name: Name of the crop
            
        Returns:
            Crop information dictionary
        """
        # This can be expanded with a crop database
        crop_info = {
            'sugarcane': {
                'name_hindi': 'गन्ना',
                'scientific_name': 'Saccharum officinarum',
                'common_diseases': ['mosaic', 'red_rot', 'rust']
            },
            'maize': {
                'name_hindi': 'मक्का',
                'scientific_name': 'Zea mays',
                'common_diseases': ['blight', 'common_rust', 'gray_leaf_spot']
            },
            'wheat': {
                'name_hindi': 'गेहूं',
                'scientific_name': 'Triticum aestivum',
                'common_diseases': ['brown_rust', 'yellow_rust', 'septoria']
            },
            'bajra': {
                'name_hindi': 'बाजरा',
                'scientific_name': 'Pennisetum glaucum',
                'common_diseases': ['downy_mildew', 'blast']
            },
            'ragi': {
                'name_hindi': 'रागी',
                'scientific_name': 'Eleusine coracana',
                'common_diseases': ['blast', 'brown_spot']
            },
            'cotton': {
                'name_hindi': 'कपास',
                'scientific_name': 'Gossypium',
                'common_diseases': ['bacterial_blight', 'curl_virus', 'fusarium_wilt']
            },
            'jute': {
                'name_hindi': 'जूट',
                'scientific_name': 'Corchorus',
                'common_diseases': ['stemrot', 'anthracnose']
            },
            'barley': {
                'name_hindi': 'जौ',
                'scientific_name': 'Hordeum vulgare',
                'common_diseases': ['net_blotch', 'scald', 'leaf_rust']
            },
            'pea': {
                'name_hindi': 'मटर',
                'scientific_name': 'Pisum sativum',
                'common_diseases': ['powdery_mildew', 'downy_mildew']
            }
        }
        
        return crop_info.get(crop_name, {})


# Global service instance
crop_service = CropClassificationService()
