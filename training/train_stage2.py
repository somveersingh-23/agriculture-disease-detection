"""
Train Stage 2: Disease Detection Models
Separate model for each crop type
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report

from training.config import Stage2Config
from training.augmentation import DataAugmentor
from training.utils import create_directories, save_training_history


class Stage2Trainer:
    """Trainer for disease detection models"""
    
    def __init__(self, crop_name, config=Stage2Config):
        self.crop_name = crop_name
        self.config = config
        self.disease_classes = config.DISEASE_CLASSES[crop_name]
        self.num_classes = len(self.disease_classes)
        self.augmentor = DataAugmentor()
        self.model = None
        self.history = None
        
        # Create necessary directories
        create_directories([
            f"{self.config.CHECKPOINT_DIR}/{crop_name}",
            self.config.MODEL_SAVE_DIR
        ])
    
    def build_model(self):
        """Build disease detection model for specific crop"""
        print(f"\nBuilding Disease Detection Model for {self.crop_name}...")
        print(f"Number of disease classes: {self.num_classes}")
        
        # Load EfficientNetB0
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.config.INPUT_SHAPE
        )
        
        # Freeze initial layers, allow fine-tuning of top layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Build model
        inputs = keras.Input(shape=self.config.INPUT_SHAPE)
        x = keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.DROPOUT_RATE)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.DROPOUT_RATE)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.config.DROPOUT_RATE * 0.5)(x)
        
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name=f'{self.crop_name}_disease_output'
        )(x)
        
        self.model = keras.Model(
            inputs,
            outputs,
            name=f'Stage2_{self.crop_name}_DiseaseDetector'
        )
        
        # Compile with class weights for imbalanced datasets
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"Model built successfully!")
        return self.model
    
    def prepare_data(self):
        """Prepare crop-specific data generators"""
        print(f"Preparing data for {self.crop_name}...")
        
        crop_data_dir = os.path.join(self.config.DATA_DIR, self.crop_name)
        
        train_gen = self.augmentor.get_train_generator()
        val_gen = self.augmentor.get_val_generator()
        
        train_data = train_gen.flow_from_directory(
            os.path.join(crop_data_dir, 'train'),
            target_size=self.config.INPUT_SHAPE[:2],
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        val_data = val_gen.flow_from_directory(
            os.path.join(crop_data_dir, 'validation'),
            target_size=self.config.INPUT_SHAPE[:2],
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        print(f"Training samples: {train_data.samples}")
        print(f"Validation samples: {val_data.samples}")
        
        return train_data, val_data
    
    def get_callbacks(self):
        """Training callbacks"""
        return [
            ModelCheckpoint(
                filepath=os.path.join(
                    self.config.CHECKPOINT_DIR,
                    self.crop_name,
                    f'{self.crop_name}_disease_best.h5'
                ),
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
        print(f"\n{'='*60}")
        print(f"TRAINING DISEASE DETECTOR: {self.crop_name.upper()}")
        print(f"{'='*60}\n")
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.EPOCHS,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        return self.history
    
    def evaluate(self):
        """Evaluate on test set"""
        print(f"\nEvaluating {self.crop_name} disease model...")
        
        test_gen = self.augmentor.get_test_generator()
        test_data = test_gen.flow_from_directory(
            os.path.join(self.config.DATA_DIR, self.crop_name, 'test'),
            target_size=self.config.INPUT_SHAPE[:2],
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        results = self.model.evaluate(test_data, verbose=1)
        print(f"\nTest Accuracy: {results[1]*100:.2f}%")
        
        # Predictions
        predictions = self.model.predict(test_data, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_data.classes
        
        print("\nClassification Report:")
        print(classification_report(
            y_true,
            y_pred,
            target_names=self.disease_classes
        ))
        
        return results
    
    def save_model(self):
        """Save disease model"""
        model_path = os.path.join(
            self.config.MODEL_SAVE_DIR,
            f'{self.crop_name}_disease.h5'
        )
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'crop_name': self.crop_name,
            'disease_classes': self.disease_classes,
            'num_classes': self.num_classes,
            'architecture': self.config.BASE_MODEL,
            'input_shape': self.config.INPUT_SHAPE
        }
        
        metadata_path = f'app/models/metadata/{self.crop_name}_metadata.json'
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)


def train_all_crops():
    """Train disease detection models for all crops"""
    crops = list(Stage2Config.DISEASE_CLASSES.keys())
    
    for crop in crops:
        print(f"\n\n{'#'*80}")
        print(f"# TRAINING: {crop.upper()}")
        print(f"{'#'*80}\n")
        
        trainer = Stage2Trainer(crop)
        trainer.build_model()
        train_data, val_data = trainer.prepare_data()
        trainer.train(train_data, val_data)
        trainer.evaluate()
        trainer.save_model()
        
        # Save history
        save_training_history(
            trainer.history,
            f'stage2_{crop}_training_history.json'
        )


if __name__ == '__main__':
    train_all_crops()
