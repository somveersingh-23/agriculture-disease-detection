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


class TrueDivide(tf.keras.layers.Layer):
    """
    Lightweight replacement for unknown 'TrueDivide' layer encountered
    when loading some .h5 models. It simply divides input by a constant.
    This helps deserialize models that used a TF op/lambda for division.
    """
    def __init__(self, divisor: float = 255.0, **kwargs):
        super().__init__(**kwargs)
        self.divisor = float(divisor)

    def call(self, inputs):
        return tf.math.truediv(inputs, tf.constant(self.divisor, dtype=inputs.dtype))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"divisor": self.divisor})
        return cfg


class ModelLoader:
    """Singleton class to load and manage ML models"""

    _instance = None
    _models_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Use instance attribute to mark loaded state so tests / multiple instances behave
        if not getattr(self, "_initialized", False):
            self.stage1_model = None
            self.stage2_models = {}
            self.crop_metadata = {}
            self.disease_metadata = {}
            self.crop_classes = []
            self.disease_classes = {}
            self._initialized = True
            self._models_loaded = False

    # ---------------------- NEW CODE: download helper ----------------------
    def _download_file(self, url: str, dest_path: str, timeout: int = 60):
        """Download file from a URL if it doesn't exist"""
        if os.path.exists(dest_path):
            logger.info(f"File already exists: {dest_path}")
            return

        logger.info(f"Downloading from {url} ...")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # stream download
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
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
            logger.error(f"Error loading models: {str(e)}", exc_info=True)
            raise

    def _load_stage1_model(self):
        """Load crop classification model"""
        logger.info(f"Loading Stage 1 model from {settings.STAGE1_MODEL_PATH}")

        # Auto-download from Hugging Face if missing
        if not os.path.exists(settings.STAGE1_MODEL_PATH):
            logger.warning("Stage 1 model not found locally. Downloading...")

            # allow settings.HF_REPO_ID override, otherwise use the known repo
            repo_id = getattr(settings, "HF_REPO_ID", "somveersingh-23/agriculture-disease-detection")
            # Try a couple of commonly used filenames (the repo seems to have mobilenetv2 variant)
            candidate_filenames = [
                "stage1_crop_classifier.h5",
                "stage1_crop_classifier_mobilenetv2.h5",
                "stage1_crop_classifier_mobilenet_v2.h5",
            ]

            downloaded = False
            for fn in candidate_filenames:
                hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{fn}"
                try:
                    self._download_file(hf_url, settings.STAGE1_MODEL_PATH)
                    downloaded = True
                    break
                except requests.HTTPError as he:
                    logger.debug(f"Stage1 candidate not found at {hf_url}: {he}")
                    # try next candidate
                except Exception as e:
                    logger.warning(f"Failed to download {hf_url}: {e}", exc_info=True)
            if not downloaded:
                raise FileNotFoundError("Stage 1 model not found locally and could not be downloaded from Hugging Face. "
                                        "Please upload the model to the repo or put it into the app/models path.")

        # Try to load the model; if unknown layer 'TrueDivide' appears, register a fallback
        try:
            self.stage1_model = tf.keras.models.load_model(
                settings.STAGE1_MODEL_PATH,
                compile=False
            )
            logger.info("Stage 1 model loaded successfully")
        except ValueError as e:
            # detect unknown layer error and attempt to recover
            msg = str(e)
            if "Unknown layer" in msg and "TrueDivide" in msg:
                logger.warning("Unknown layer 'TrueDivide' detected while loading Stage 1 model. "
                               "Attempting to register fallback custom layer and retry loading.")
                custom_objs = {"TrueDivide": TrueDivide}
                # try again with custom_objects
                self.stage1_model = tf.keras.models.load_model(
                    settings.STAGE1_MODEL_PATH,
                    compile=False,
                    custom_objects=custom_objs
                )
                logger.info("Stage 1 model loaded successfully with custom_objects fallback")
            else:
                logger.error("Failed to load Stage 1 model", exc_info=True)
                raise

    def _load_stage2_models(self):
        """Load all disease detection models"""
        logger.info(f"Loading Stage 2 models from {settings.STAGE2_MODELS_DIR}")

        if not os.path.exists(settings.STAGE2_MODELS_DIR):
            logger.warning("Stage 2 model directory missing. Creating...")
            os.makedirs(settings.STAGE2_MODELS_DIR, exist_ok=True)

        # If the settings or repo contains HF repo id, use it; fallback to your repo
        repo_id = getattr(settings, "HF_REPO_ID", "somveersingh-23/agriculture-disease-detection")
        # map of crop -> filename (these are the filenames you showed on HF)
        stage2_model_files = {
            "bajra": "bajra_disease.h5",
            "cotton": "cotton_disease.h5",
            "jute": "jute_disease.h5",
            "maize": "maize_disease.h5",
            "pea": "pea_disease.h5",
            "ragi": "ragi_disease.h5",
            "rice": "rice_disease.h5",
            "sugarcane": "sugarcane_disease.h5",
            "wheat": "wheat_disease.h5",
        }

        for crop_name, filename in stage2_model_files.items():
            model_path = os.path.join(settings.STAGE2_MODELS_DIR, f"{crop_name}_disease.h5")
            hf_url = f"https://huggingface.co/{repo_id}/resolve/main/stage2_disease_models/{filename}"

            # Auto-download each Stage 2 model if missing
            if not os.path.exists(model_path):
                logger.warning(f"{crop_name} model missing locally. Downloading from Hugging Face...")
                try:
                    self._download_file(hf_url, model_path)
                except requests.HTTPError as he:
                    logger.warning(f"{crop_name} model not found at {hf_url}: {he}")
                    # continue to next crop rather than crash — absence of some crop models may be acceptable
                    continue
                except Exception as e:
                    logger.error(f"Failed to download {crop_name} model: {e}", exc_info=True)
                    continue

            # If file exists after attempted download, load it
            if os.path.exists(model_path):
                try:
                    logger.info(f"Loading {crop_name} disease model...")
                    # same fallback for TrueDivide as stage1
                    try:
                        self.stage2_models[crop_name] = tf.keras.models.load_model(
                            model_path,
                            compile=False
                        )
                    except ValueError as e:
                        msg = str(e)
                        if "Unknown layer" in msg and "TrueDivide" in msg:
                            logger.warning(f"Unknown layer 'TrueDivide' in {crop_name} model — using fallback custom layer.")
                            custom_objs = {"TrueDivide": TrueDivide}
                            self.stage2_models[crop_name] = tf.keras.models.load_model(
                                model_path,
                                compile=False,
                                custom_objects=custom_objs
                            )
                        else:
                            raise
                except Exception as e:
                    logger.error(f"Failed to load {crop_name} model: {e}", exc_info=True)

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
        return self.disease_classes.get(crop_name, {})


# Global instance
model_loader = ModelLoader()
