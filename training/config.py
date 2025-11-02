"""
Updated Training Configuration
- Uses MobileNetV2 (lightweight + fast)
- Includes dataset balancing options
- Compatible with Stage 1 & Stage 2 pipelines
"""

import os
from pathlib import Path

class BaseConfig:
    """Base configuration"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    MODEL_DIR = PROJECT_ROOT / 'app' / 'models'
    CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
    LOG_DIR = PROJECT_ROOT / 'logs'
    
    # Create directories if missing
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # ✅ Model architecture (changed to MobileNetV2)
    BASE_MODEL = 'MobileNetV2'  
    INPUT_SHAPE = (224, 224, 3)
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0005  # 🔽 Slightly lower for fine-tuning
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    
    # ✅ Data balancing
    USE_CLASS_WEIGHTS = True  # enable compute_class_weight()
    MIN_SAMPLES_PER_CLASS = 1000  # target per class after augmentation
    
    # Data split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    RANDOM_SEED = 42


class Stage1Config(BaseConfig):
    """Stage 1: Crop Classification Config"""
    
    MODEL_NAME = 'stage1_crop_classifier_mobilenetv2'
    DATA_PATH = BaseConfig.DATA_DIR / 'processed' / 'stage1_crops'
    MODEL_SAVE_PATH = BaseConfig.MODEL_DIR / 'stage1_crop_classifier_mobilenetv2.h5'
    CHECKPOINT_PATH = BaseConfig.CHECKPOINT_DIR / 'stage1'
    
    DROPOUT_RATE = 0.3
    
    # ✅ Include 'rice' explicitly and maintain consistent order
    CROP_CLASSES = [
        'sugarcane', 'maize', 'wheat', 'bajra',
        'ragi', 'cotton', 'jute', 'pea', 'rice'
    ]
    NUM_CLASSES = len(CROP_CLASSES)


class Stage2Config(BaseConfig):
    """Stage 2: Disease Detection Config"""
    
    MODEL_NAME_PREFIX = 'stage2_disease_mobilenetv2'
    DATA_PATH = BaseConfig.DATA_DIR / 'processed' / 'stage2_diseases'
    MODEL_SAVE_DIR = BaseConfig.MODEL_DIR / 'stage2_disease_models'
    CHECKPOINT_PATH = BaseConfig.CHECKPOINT_DIR / 'stage2'
    
    DROPOUT_RATE = 0.4
    
    # Each crop’s diseases will be dynamically loaded
    DISEASE_CLASSES = {}


class AugmentationConfig:
    """Data Augmentation Configuration"""
    
    # ✅ Stronger augmentations for small classes
    ROTATION_RANGE = 45
    WIDTH_SHIFT_RANGE = 0.25
    HEIGHT_SHIFT_RANGE = 0.25
    SHEAR_RANGE = 0.2
    ZOOM_RANGE = 0.3
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = True
    BRIGHTNESS_RANGE = [0.7, 1.3]
    FILL_MODE = 'nearest'
    
    USE_MIXUP = True
    USE_CUTMIX = False
    MIXUP_ALPHA = 0.2
