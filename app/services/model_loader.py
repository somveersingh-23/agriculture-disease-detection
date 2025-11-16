"""
Model loading and management service
- Downloads models (Hugging Face) if missing
- Robustly loads legacy .h5 models under Keras 3
- Avoids Keras positional-argument deserialization errors by:
  * registering custom objects for Lambda/TFOp ops
  * rebuilding model and loading weights if needed
- Supports lazy loading via env: LOAD_MODELS_ON_STARTUP=lazy
"""

import os
import io
import json
import logging
import requests
from typing import Dict, Any, Optional

import tensorflow as tf
from tensorflow import keras

from app.config import settings

logger = logging.getLogger(__name__)


# --------------------- Safe shims for legacy Lambda/TFOp ---------------------

class TrueDivide(keras.layers.Layer):
    """Fallback for serialized 'TrueDivide' op. Prefer preprocessing outside model."""
    def __init__(self, divisor: float = 255.0, **kwargs):
        super().__init__(**kwargs)
        self.divisor = float(divisor)
    def call(self, inputs, divisor=None):
        x = tf.cast(inputs, tf.float32)
        d = tf.cast(divisor if divisor is not None else self.divisor, tf.float32)
        return tf.math.divide(x, d)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"divisor": self.divisor})
        return cfg

class SubtractConst(keras.layers.Layer):
    """Fallback for subtraction Lambda like x - 1.0."""
    def __init__(self, value: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.value = float(value)
    def call(self, inputs, value=None):
        x = tf.cast(inputs, tf.float32)
        v = tf.cast(value if value is not None else self.value, tf.float32)
        return tf.math.subtract(x, v)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"value": self.value})
        return cfg

class SafeLambda(keras.layers.Layer):
    """
    Very generic fallback for unknown Lambda/TFOpLambda nodes.
    No-ops by default, intended only to let model load instead of crash.
    Move real preprocessing to app.services.image_preprocessing.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs, **kwargs):
        return tf.identity(inputs)
    def get_config(self):
        return super().get_config()

# Alias some common serialized names seen in H5 configs
CUSTOM_OBJECTS = {
    "TrueDivide": TrueDivide,
    "Divide": TrueDivide,
    "Sub": SubtractConst,
    "Subtract": SubtractConst,
    "TFOpLambda": SafeLambda,
    "Lambda": SafeLambda,
}


def _custom_scope():
    """Context manager to register custom objects during load_model."""
    return keras.utils.custom_object_scope(CUSTOM_OBJECTS)


# ----------------------------- Download helper ------------------------------

def _download_file(url: str, dest_path: str, timeout: int = 90):
    if os.path.exists(dest_path):
        logger.info(f"File present: {dest_path}")
        return
    logger.info(f"Downloading: {url}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
    logger.info(f"Saved to: {dest_path}")


# --------------------------- Rebuild if necessary ---------------------------

def _build_stage1_skeleton(num_classes: int = 9, input_shape=(224, 224, 3)) -> keras.Model:
    base = keras.applications.EfficientNetB0(
        include_top=False, input_shape=input_shape, weights=None
    )
    x = keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.Dropout(0.30, name="drop1")(x)
    x = keras.layers.Dense(512, activation="relu", name="fc1")(x)
    x = keras.layers.BatchNormalization(name="bn2")(x)
    x = keras.layers.Dropout(0.30, name="drop2")(x)
    x = keras.layers.Dense(256, activation="relu", name="fc2")(x)
    x = keras.layers.Dropout(0.15, name="drop3")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="crop_output")(x)
    model = keras.Model(base.input, outputs, name="Stage1_Rebuilt")
    return model


def _try_load_model(model_path: str) -> keras.Model:
    """
    Attempt multiple strategies to load:
    1) Direct load (SavedModel/.keras/.h5)
    2) Direct with custom_objects scope (for Lambda/TFOp)
    3) Rebuild skeleton + load_weights with skip_mismatch
    """
    # Strategy 1: direct load
    try:
        with _custom_scope():
            m = keras.models.load_model(model_path, compile=False)
        return m
    except Exception as e1:
        logger.warning(f"Direct load failed: {e1}")

    # Strategy 2: explicit scope again (some environments differ)
    try:
        with _custom_scope():
            m = keras.models.load_model(model_path, compile=False, custom_objects=CUSTOM_OBJECTS)
        return m
    except Exception as e2:
        logger.warning(f"Scoped load failed: {e2}")

    # Strategy 3: rebuild and load weights
    logger.info("Rebuilding stage1 skeleton and loading weights with skip_mismatch...")
    m = _build_stage1_skeleton()
    try:
        m.load_weights(model_path, by_name=True, skip_mismatch=True)
        return m
    except Exception as e3:
        raise RuntimeError(f"Failed to recover model from weights: {e3}")


# ------------------------------- ModelLoader --------------------------------

class ModelLoader:
    """Singleton loader with lazy-load support for Render."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.stage1_model: Optional[keras.Model] = None
        self.stage2_models: Dict[str, keras.Model] = {}
        self.crop_metadata: Dict[str, Any] = {}
        self.disease_metadata: Dict[str, Any] = {}
        self.crop_classes = []
        self.disease_classes = {}
        self._models_loaded = False
        self._initialized = True

    # Public API
    def load_all_models(self):
        """Load models at startup unless LOAD_MODELS_ON_STARTUP=lazy."""
        if self._models_loaded:
            logger.info("Models already loaded")
            return

        if os.getenv("LOAD_MODELS_ON_STARTUP", "").lower() == "lazy":
            logger.info("Lazy loading enabled. Models will load on first request.")
            self._models_loaded = True  # mark OK; actual models load when accessed
            return

        logger.info("Starting model loading process...")
        self._ensure_stage1()
        self._ensure_stage2_all()
        self._load_metadata()
        self._models_loaded = True
        logger.info("✅ All models loaded successfully.")

    def get_stage1_model(self):
        if self.stage1_model is None:
            self._ensure_stage1()
        return self.stage1_model

    def get_stage2_model(self, crop_name: str):
        if crop_name not in self.stage2_models:
            self._ensure_stage2(crop_name)
        return self.stage2_models[crop_name]

    def get_crop_classes(self):
        if not self.crop_classes:
            self._load_metadata()
        return self.crop_classes

    def get_disease_classes(self, crop_name: str):
        if crop_name not in self.disease_classes:
            self._load_metadata()
        return self.disease_classes.get(crop_name, [])

    # Internals
    def _ensure_stage1(self):
        path = settings.STAGE1_MODEL_PATH
        logger.info(f"Loading Stage 1 from {path}")

        if not os.path.exists(path):
            logger.warning("Stage 1 model not found locally. Downloading...")
            repo_id = getattr(settings, "HF_REPO_ID", "somveersingh-23/agriculture-disease-detection")
            candidates = [
                "stage1_crop_classifier.keras",
                "stage1_crop_classifier",           # SavedModel dir (if uploaded)
                "stage1_crop_classifier.h5",
                "stage1_crop_classifier_mobilenetv2.h5",
                "stage1_crop_classifier_mobilenet_v2.h5",
            ]
            ok = False
            for fn in candidates:
                url = f"https://huggingface.co/{repo_id}/resolve/main/{fn}"
                try:
                    _download_file(url, path)
                    ok = True
                    break
                except Exception as e:
                    logger.debug(f"Stage1 candidate not found at {url}: {e}")
            if not ok:
                raise FileNotFoundError("Stage 1 model not available locally or on HF.")

        # Try to load safely
        self.stage1_model = _try_load_model(path)
        logger.info("Stage 1 model ready.")

    def _ensure_stage2(self, crop_name: str):
        os.makedirs(settings.STAGE2_MODELS_DIR, exist_ok=True)
        model_path = os.path.join(settings.STAGE2_MODELS_DIR, f"{crop_name}_disease.h5")
        # Prefer modern formats if you upload later
        modern_path = os.path.join(settings.STAGE2_MODELS_DIR, f"{crop_name}_disease.keras")
        if os.path.exists(modern_path):
            model_path = modern_path

        if not os.path.exists(model_path):
            repo_id = getattr(settings, "HF_REPO_ID", "somveersingh-23/agriculture-disease-detection")
            hf_candidates = [
                f"stage2_disease_models/{crop_name}_disease.keras",
                f"stage2_disease_models/{crop_name}_disease.h5",
            ]
            logger.warning(f"{crop_name} model missing. Downloading from HF...")
            ok = False
            for fn in hf_candidates:
                url = f"https://huggingface.co/{repo_id}/resolve/main/{fn}"
                try:
                    _download_file(url, model_path)
                    ok = True
                    break
                except Exception as e:
                    logger.debug(f"{crop_name} not found at {url}: {e}")
            if not ok:
                logger.warning(f"Skipping {crop_name}: model not found on HF.")
                return

        # Load safely
        try:
            with _custom_scope():
                self.stage2_models[crop_name] = keras.models.load_model(model_path, compile=False)
        except Exception:
            # Fallback: rebuild a classifier head matching metadata if available
            try:
                classes = self.get_disease_classes(crop_name)
                num = len(classes) if classes else 3
                base = keras.applications.EfficientNetB0(include_top=False, input_shape=(224,224,3), weights=None)
                x = keras.layers.GlobalAveragePooling2D()(base.output)
                x = keras.layers.Dropout(0.4)(x)
                x = keras.layers.Dense(512, activation='relu')(x)
                x = keras.layers.Dropout(0.4)(x)
                out = keras.layers.Dense(num, activation='softmax', name=f'{crop_name}_disease_output')(x)
                m = keras.Model(base.input, out)
                m.load_weights(model_path, by_name=True, skip_mismatch=True)
                self.stage2_models[crop_name] = m
            except Exception as e2:
                logger.error(f"Failed to load or rebuild {crop_name}: {e2}", exc_info=True)

    def _ensure_stage2_all(self):
        # Optionally pre-load a known set; otherwise lazy-load on demand.
        preload = os.getenv("PRELOAD_STAGE2", "false").lower() == "true"
        if not preload:
            logger.info("Stage 2 models will be loaded lazily per crop.")
            return
        # If you want to preload a fixed set:
        crops = ["bajra","cotton","jute","maize","pea","ragi","sugarcane","wheat","barley"]
        for c in crops:
            self._ensure_stage2(c)

    def _load_metadata(self):
        logger.info("Loading model metadata...")
        crop_meta = os.path.join(settings.METADATA_DIR, "crop_metadata.json")
        if os.path.exists(crop_meta):
            with open(crop_meta, "r", encoding="utf-8") as f:
                self.crop_metadata = json.load(f)
                self.crop_classes = self.crop_metadata.get("crop_classes", [])
        # Disease metadata per crop
        for fname in os.listdir(settings.METADATA_DIR) if os.path.exists(settings.METADATA_DIR) else []:
            if not fname.endswith("_metadata.json") or fname == "crop_metadata.json":
                continue
            crop = fname.replace("_metadata.json", "")
            with open(os.path.join(settings.METADATA_DIR, fname), "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.disease_metadata[crop] = meta
                self.disease_classes[crop] = meta.get("disease_classes", [])
        logger.info("Metadata loaded.")


# Global instance
model_loader = ModelLoader()
