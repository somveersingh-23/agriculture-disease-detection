"""
Model loading and management service
Loads models at startup for efficient inference
"""
import os
import json
import logging
import tensorflow as tf
from typing import Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton class to load and manage ML models"""
    
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._models_loaded:
            self.stage1_model = None
            self.stage2_models = {}
            self.crop_metadata = {}
            self.disease_metadata = {}
            self.crop_classes = []
            self.disease_classes = {}
            
    def load_all_models(self):
        """Load all models at application startup"""
        if self._models_loaded:
            logger.info("Models already loaded")
            return
        
        try:
            logger.info("Starting model loading process...")
            
            # Load Stage 1: Crop Classifier
            self._load_stage1_model()
            
            # Load Stage 2: Disease Models
            self._load_stage2_models()
            
            # Load metadata
            self._load_metadata()
            
            self._models_loaded = True
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _load_stage1_model(self):
        """Load crop classification model"""
        logger.info(f"Loading Stage 1 model from {settings.STAGE1_MODEL_PATH}")
        
        if not os.path.exists(settings.STAGE1_MODEL_PATH):
            raise FileNotFoundError(f"Stage 1 model not found at {settings.STAGE1_MODEL_PATH}")
        
        self.stage1_model = tf.keras.models.load_model(
            settings.STAGE1_MODEL_PATH,
            compile=False
        )
        logger.info("Stage 1 model loaded successfully")
    
    def _load_stage2_models(self):
        """Load all disease detection models"""
        logger.info(f"Loading Stage 2 models from {settings.STAGE2_MODELS_DIR}")
        
        if not os.path.exists(settings.STAGE2_MODELS_DIR):
            raise FileNotFoundError(f"Stage 2 models directory not found")
        
        # Get list of disease model files
        model_files = [f for f in os.listdir(settings.STAGE2_MODELS_DIR) 
                      if f.endswith('_disease.h5')]
        
        for model_file in model_files:
            crop_name = model_file.replace('_disease.h5', '')
            model_path = os.path.join(settings.STAGE2_MODELS_DIR, model_file)
            
            logger.info(f"Loading {crop_name} disease model...")
            self.stage2_models[crop_name] = tf.keras.models.load_model(
                model_path,
                compile=False
            )
        
        logger.info(f"Loaded {len(self.stage2_models)} disease detection models")
    
    def _load_metadata(self):
        """Load model metadata"""
        logger.info("Loading model metadata...")
        
        # Load crop metadata
        crop_metadata_path = os.path.join(settings.METADATA_DIR, "crop_metadata.json")
        if os.path.exists(crop_metadata_path):
            with open(crop_metadata_path, 'r', encoding='utf-8') as f:
                self.crop_metadata = json.load(f)
                self.crop_classes = self.crop_metadata.get('crop_classes', [])
        
        # Load disease metadata for each crop
        for crop in self.stage2_models.keys():
            metadata_path = os.path.join(settings.METADATA_DIR, f"{crop}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.disease_metadata[crop] = metadata
                    self.disease_classes[crop] = metadata.get('disease_classes', [])
        
        logger.info("Metadata loaded successfully")
    
    def get_stage1_model(self):
        """Get crop classification model"""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_all_models() first")
        return self.stage1_model
    
    def get_stage2_model(self, crop_name: str):
        """Get disease detection model for specific crop"""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_all_models() first")
        
        if crop_name not in self.stage2_models:
            raise ValueError(f"No disease model found for crop: {crop_name}")
        
        return self.stage2_models[crop_name]
    
    def get_crop_classes(self):
        """Get list of crop classes"""
        return self.crop_classes
    
    def get_disease_classes(self, crop_name: str):
        """Get disease classes for specific crop"""
        return self.disease_classes.get(crop_name, [])


# Global model loader instance
model_loader = ModelLoader()
