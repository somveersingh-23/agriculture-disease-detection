"""
Model loading and management service
Automatically downloads models from Hugging Face if missing.
"""

import os
import json
import logging
import requests
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

    # ---------------------- NEW CODE: download helper ----------------------
    def _download_file(self, url: str, dest_path: str):
        """Download file from a URL if it doesn't exist"""
        if os.path.exists(dest_path):
            logger.info(f"File already exists: {dest_path}")
            return

        logger.info(f"Downloading from {url} ...")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded and saved to {dest_path}")
    # ----------------------------------------------------------------------

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
            logger.info("✅ All models loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def _load_stage1_model(self):
        """Load crop classification model"""
        logger.info(f"Loading Stage 1 model from {settings.STAGE1_MODEL_PATH}")

        # 🧠 Auto-download from Hugging Face if missing
        if not os.path.exists(settings.STAGE1_MODEL_PATH):
            logger.warning("Stage 1 model not found locally. Downloading...")
            hf_url = "https://huggingface.co/somveersingh-23/agriculture-disease-detection/resolve/main/stage1_crop_classifier_mobilenetv2.h5"
            self._download_file(hf_url, settings.STAGE1_MODEL_PATH)

        self.stage1_model = tf.keras.models.load_model(
            settings.STAGE1_MODEL_PATH,
            compile=False
        )
        logger.info("Stage 1 model loaded successfully")

    def _load_stage2_models(self):
        """Load all disease detection models"""
        logger.info(f"Loading Stage 2 models from {settings.STAGE2_MODELS_DIR}")

        if not os.path.exists(settings.STAGE2_MODELS_DIR):
            logger.warning("Stage 2 model directory missing. Creating...")
            os.makedirs(settings.STAGE2_MODELS_DIR, exist_ok=True)

        # Example: list of crops and their model URLs (you can extend)
        stage2_model_urls = {
             "bajra": "https://huggingface.co/somveersingh-23/agriculture-disease-detection/resolve/main/stage2_disease_models/bajra_disease.h5",
            "cotton": "https://huggingface.co/somveersingh-23/agriculture-disease-detection/resolve/main/stage2_disease_models/cotton_disease.h5",
    "jute": "https://huggingface.co/somveersingh-23/agriculture-disease-detection/resolve/main/stage2_disease_models/jute_disease.h5",
    "maize": "https://huggingface.co/somveersingh-23/agriculture-disease-detection/resolve/main/stage2_disease_models/maize_disease.h5",
    "pea": "https://huggingface.co/somveersingh-23/agriculture-disease-detection/resolve/main/stage2_disease_models/pea_disease.h5",
    "ragi": "https://huggingface.co/somveersingh-23/agriculture-disease-detection/resolve/main/stage2_disease_models/ragi_disease.h5",
    "rice": "https://huggingface.co/somveersingh-23/agriculture-disease-detection/resolve/main/stage2_disease_models/rice_disease.h5",
    "sugarcane": "https://huggingface.co/somveersingh-23/agriculture-disease-detection/resolve/main/stage2_disease_models/sugarcane_disease.h5",
    "wheat": "https://huggingface.co/somveersingh-23/agriculture-disease-detection/resolve/main/stage2_disease_models/wheat_disease.h5"
        }

        for crop_name, url in stage2_model_urls.items():
            model_path = os.path.join(settings.STAGE2_MODELS_DIR, f"{crop_name}_disease.h5")

            # 🧠 Auto-download each Stage 2 model
            if not os.path.exists(model_path):
                logger.warning(f"{crop_name} model missing. Downloading...")
                self._download_file(url, model_path)

            logger.info(f"Loading {crop_name} disease model...")
            self.stage2_models[crop_name] = tf.keras.models.load_model(
                model_path,
                compile=False
            )

        logger.info(f"Loaded {len(self.stage2_models)} disease detection models")

    def _load_metadata(self):
        """Load model metadata"""
        logger.info("Loading model metadata...")

        crop_metadata_path = os.path.join(settings.METADATA_DIR, "crop_metadata.json")
        if os.path.exists(crop_metadata_path):
            with open(crop_metadata_path, 'r', encoding='utf-8') as f:
                self.crop_metadata = json.load(f)
                self.crop_classes = self.crop_metadata.get('crop_classes', [])

        for crop in self.stage2_models.keys():
            metadata_path = os.path.join(settings.METADATA_DIR, f"{crop}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.disease_metadata[crop] = metadata
                    self.disease_classes[crop] = metadata.get('disease_classes', [])

        logger.info("Metadata loaded successfully")

    def get_stage1_model(self):
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_all_models() first")
        return self.stage1_model

    def get_stage2_model(self, crop_name: str):
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_all_models() first")
        if crop_name not in self.stage2_models:
            raise ValueError(f"No disease model found for crop: {crop_name}")
        return self.stage2_models[crop_name]

    def get_crop_classes(self):
        return self.crop_classes

    def get_disease_classes(self, crop_name: str):
        return self.disease_classes.get(crop_name, [])


# Global instance
model_loader = ModelLoader()
 