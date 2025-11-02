"""
Training Script for Stage 2: Disease Detection
Trains separate models for each crop
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import json
from datetime import datetime

from training.config import Stage2Config
from training.augmentation import DataAugmentor
from training.utils import create_directories, save_training_history, plot_training_history

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report


class Stage2Trainer:
    """Stage 2: Disease Detection Trainer"""
    
    def __init__(self, crop_name):
        self.crop_name = crop_name
        self.config = Stage2Config()
        self.augmentor = DataAugmentor()
        self.model = None
        self.history = None
        self.disease_classes = []
        
        # Set random seeds
        np.random.seed(self.config.RANDOM_SEED)
        tf.random.set_seed(self.config.RANDOM_SEED)
        
        # Create directories
        create_directories([
            self.config.CHECKPOINT_PATH / crop_name,
            self.config.MODEL_SAVE_DIR,
            self.config.LOG_DIR
        ])
        
        print("\n" + "="*70)
        print(f" "*10 + f"STAGE 2: DISEASE DETECTION - {crop_name.upper()}")
        print("="*70)
    
    def build_model(self, num_classes):
        """Build disease detection model"""
        print(f"\n[1/5] Building Model for {self.crop_name}...")
        print(f"Number of disease classes: {num_classes}")
        
        # Load EfficientNetB0
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.config.INPUT_SHAPE
        )
        
        # Fine-tune top layers only
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Build model
        inputs = keras.Input(shape=self.config.INPUT_SHAPE)
        x = keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=True)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.DROPOUT_RATE)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.DROPOUT_RATE)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.config.DROPOUT_RATE * 0.5)(x)
        
        outputs = layers.Dense(
            num_classes,
            activation='softmax',
            name=f'{self.crop_name}_disease_output'
        )(x)
        
        self.model = keras.Model(inputs, outputs, name=f'{self.crop_name}_DiseaseDetector')
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"✓ Model built: {self.model.count_params():,} parameters")
        return self.model
    
    def prepare_data(self):
        """Prepare data for specific crop"""
        print(f"\n[2/5] Loading {self.crop_name} disease data...")
        
        crop_data_path = self.config.DATA_PATH / self.crop_name
        
        if not crop_data_path.exists():
            raise FileNotFoundError(f"Data not found: {crop_data_path}")
        
        train_gen = self.augmentor.get_train_generator()
        val_gen = self.augmentor.get_val_generator()
        
        train_data = train_gen.flow_from_directory(
            str(crop_data_path / 'train'),
            target_size=self.config.INPUT_SHAPE[:2],
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=self.config.RANDOM_SEED
        )
        
        val_data = val_gen.flow_from_directory(
            str(crop_data_path / 'validation'),
            target_size=self.config.INPUT_SHAPE[:2],
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        self.disease_classes = list(train_data.class_indices.keys())
        num_classes = len(self.disease_classes)
        
        print(f"✓ Data loaded!")
        print(f"  Training samples: {train_data.samples}")
        print(f"  Validation samples: {val_data.samples}")
        print(f"  Disease classes: {', '.join(self.disease_classes)}")
        
        return train_data, val_data, num_classes
    
    def get_callbacks(self):
        """Setup callbacks"""
        return [
            ModelCheckpoint(
                filepath=str(self.config.CHECKPOINT_PATH / self.crop_name / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            )
        ]
    
    def train(self, train_data, val_data):
        """Train disease detection model"""
        print(f"\n[3/5] Training {self.crop_name} disease model...")
        
        callbacks = self.get_callbacks()
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Training completed!")
        return self.history
    
    def evaluate(self):
        """Evaluate on test set"""
        print(f"\n[4/5] Evaluating {self.crop_name} model...")
        
        test_gen = self.augmentor.get_test_generator()
        test_data = test_gen.flow_from_directory(
            str(self.config.DATA_PATH / self.crop_name / 'test'),
            target_size=self.config.INPUT_SHAPE[:2],
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        results = self.model.evaluate(test_data, verbose=1)
        
        print("\n" + "="*70)
        print(f"TEST RESULTS - {self.crop_name.upper()}")
        print("="*70)
        print(f"Accuracy: {results[1]*100:.2f}%")
        print(f"Precision: {results[2]*100:.2f}%")
        print(f"Recall: {results[3]*100:.2f}%")
        print("="*70)
        
        # Predictions
        predictions = self.model.predict(test_data, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_data.classes
        
        print("\nClassification Report:")
        print(classification_report(
            y_true,
            y_pred,
            target_names=self.disease_classes,
            digits=4
        ))
        
        return results
    
    def save_model(self):
        """Save model and metadata"""
        print(f"\n[5/5] Saving {self.crop_name} model...")
        
        # Save model
        model_path = self.config.MODEL_SAVE_DIR / f'{self.crop_name}_disease.h5'
        self.model.save(str(model_path))
        print(f"✓ Model saved: {model_path}")
        
        # Save metadata
        metadata = {
            'crop_name': self.crop_name,
            'disease_classes': self.disease_classes,
            'num_classes': len(self.disease_classes),
            'architecture': self.config.BASE_MODEL,
            'input_shape': list(self.config.INPUT_SHAPE),
            'training_date': datetime.now().isoformat()
        }
        
        metadata_path = self.config.MODEL_DIR / 'metadata' / f'{self.crop_name}_metadata.json'
        metadata_path.parent.mkdir(exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {metadata_path}")
        
        # Save history
        history_path = self.config.LOG_DIR / f'stage2_{self.crop_name}_history.json'
        save_training_history(self.history, str(history_path))
        
        # Plot
        plot_path = self.config.LOG_DIR / f'stage2_{self.crop_name}_curves.png'
        plot_training_history(self.history, str(plot_path))
        print(f"✓ Training curves saved: {plot_path}")
    
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        try:
            # Prepare data
            train_data, val_data, num_classes = self.prepare_data()
            
            # Build model
            self.build_model(num_classes)
            
            # Train
            self.train(train_data, val_data)
            
            # Evaluate
            self.evaluate()
            
            # Save
            self.save_model()
            
            print(f"\n✓ {self.crop_name} training completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Error training {self.crop_name}: {str(e)}")
            import traceback
            traceback.print_exc()


def train_all_crops():
    """Train disease models for all crops"""
    
    print("\n" + "="*70)
    print(" "*15 + "STAGE 2: ALL CROPS DISEASE DETECTION")
    print("="*70)
    
    # Get all crop directories
    data_path = Path('data/processed/stage2_diseases')
    
    if not data_path.exists():
        print(f"✗ Data directory not found: {data_path}")
        print("Please run: python scripts/02_prepare_data.py")
        return
    
    crop_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    crops = [d.name for d in crop_dirs]
    
    print(f"\nFound {len(crops)} crops to train:")
    for i, crop in enumerate(crops, 1):
        print(f"  {i}. {crop}")
    
    print(f"\nEstimated time: ~{len(crops) * 2} hours on GPU")
    input("\nPress Enter to start training all crops...")
    
    # Train each crop
    for i, crop in enumerate(crops, 1):
        print("\n\n" + "#"*70)
        print(f"# CROP {i}/{len(crops)}: {crop.upper()}")
        print("#"*70)
        
        trainer = Stage2Trainer(crop)
        trainer.run_full_pipeline()
    
    print("\n" + "="*70)
    print(" "*15 + "ALL CROPS TRAINING COMPLETED!")
    print("="*70)
    print("\nNext step: Deploy API to Render")


def main():
    """Main function"""
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU available: {len(gpus)} device(s)")
    else:
        print("⚠ No GPU. Training will be slow.")
    
    train_all_crops()


if __name__ == '__main__':
    main()
