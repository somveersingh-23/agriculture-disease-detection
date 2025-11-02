"""
FastAPI Application Configuration
"""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    
    # App info
    APP_NAME: str = "Agriculture Disease Detection API"
    APP_VERSION: str = "1.0.0"
    DESCRIPTION: str = "Two-stage hierarchical crop disease detection for Indian farmers"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 10000
    DEBUG: bool = False
    
    # CORS
    ALLOWED_ORIGINS: list = ["*"]  # Update for production
    
    # Model paths
    STAGE1_MODEL_PATH: str = "app/models/stage1_crop_classifier.h5"
    STAGE2_MODELS_DIR: str = "app/models/stage2_disease_models"
    METADATA_DIR: str = "app/models/metadata"
    
    # Treatment database
    TREATMENT_DB_PATH: str = "app/database/treatment_data.json"
    REGIONAL_NAMES_PATH: str = "app/database/regional_names.json"
    
    # Image processing
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/jpg", "image/png"]
    MODEL_INPUT_SIZE: tuple = (224, 224)
    
    # Prediction thresholds
    CROP_CONFIDENCE_THRESHOLD: float = 0.60
    DISEASE_CONFIDENCE_THRESHOLD: float = 0.50
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # External APIs (if using)
    PLANTIX_API_KEY: str = os.getenv("PLANTIX_API_KEY", "")
    PLANTIX_API_URL: str = "https://api.plantix.net/v1"
    
    # Language support
    SUPPORTED_LANGUAGES: list = ["en", "hi", "te", "ta", "mr", "bn"]
    DEFAULT_LANGUAGE: str = "hi"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings():
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
