"""
Train Stage 1: Crop Type Classification Model
Updated to use MobileNetV2 + Class Balancing + Improved Fine-Tuning
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from training.config import Stage1Config
from training.augmentation import DataAugmentor
from training.utils import create_directories, save_training_history


class Stage1Trainer:
    """Trainer for crop classification model"""
    
    def __init__(self, config=Stage1Config):
        self.config = config
        self.augmentor = DataAugmentor()
        self.model = None
        self.history = None
        
        create_directories([
            self.config.CHECKPOINT_DIR,
            os.path.dirname(self.config.MODEL_SAVE_PATH)
        ])
    
    def build_model(self):
        """Build MobileNetV2-based crop classifier"""
        print("Building Stage 1 Crop Classification Model (MobileNetV2)...")
        
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=self.config.INPUT_SHAPE
        )
        base_model.trainable = False  # Freeze for initial training
        
        inputs = keras.Input(shape=self.config.INPUT_SHAPE)
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.DROPOUT_RATE)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.DROPOUT_RATE * 0.5)(x)
        
        outputs = layers.Dense(
            self.config.NUM_CLASSES,
            activation='softmax',
            name='crop_output'
        )(x)
        
        self.model = keras.Model(inputs, outputs, name='Stage1_CropClassifier_MobileNetV2')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"✅ Model built successfully with {self.model.count_params():,} parameters.")
        return self.model
    
    def prepare_data(self):
        """Prepare data generators"""
        print("Preparing data generators...")
        
        train_gen = self.augmentor.get_train_generator()
        val_gen = self.augmentor.get_val_generator()
        
        train_data = train_gen.flow_from_directory(
            os.path.join(self.config.DATA_PATH, 'train'),
            target_size=self.config.INPUT_SHAPE[:2],
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        val_data = val_gen.flow_from_directory(
            os.path.join(self.config.DATA_PATH, 'validation'),
            target_size=self.config.INPUT_SHAPE[:2],
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        print(f"Training samples: {train_data.samples}")
        print(f"Validation samples: {val_data.samples}")
        print(f"Classes: {train_data.class_indices}")
        
        return train_data, val_data
    
    def compute_class_weights(self, train_data):
        """Compute class weights to handle imbalance"""
        print("Computing class weights for balancing...")
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_data.classes),
            y=train_data.classes
        )
        class_weights = dict(enumerate(class_weights))
        print(f"Class Weights: {class_weights}")
        return class_weights

    def get_callbacks(self):
        """Define training callbacks"""
        return [
            ModelCheckpoint(
                filepath=os.path.join(
                    self.config.CHECKPOINT_DIR,
                    'mobilenetv2_epoch{epoch:02d}_valacc{val_accuracy:.4f}.h5'
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
            ),
            TensorBoard(
                log_dir=os.path.join('logs', 'stage1', datetime.now().strftime("%Y%m%d-%H%M%S")),
                histogram_freq=1
            )
        ]
    
    def train(self, train_data, val_data):
        """Train model with class balancing"""
        print("\n" + "="*60)
        print("STARTING STAGE 1 TRAINING (MobileNetV2)")
        print("="*60 + "\n")
        
        # Always compute class weights for imbalanced datasets
        class_weights = self.compute_class_weights(train_data)
        
        # Phase 1: Train frozen base
        print("Phase 1: Training frozen base...")
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.EPOCHS // 2,
            callbacks=self.get_callbacks(),
            class_weight=class_weights,
            verbose=1
        )
        
        # Phase 2: Fine-tune with unfreezed layers
        print("\nPhase 2: Fine-tuning MobileNetV2 base layers...")
        self.model.layers[2].trainable = True  # Unfreeze base
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        history_finetune = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.EPOCHS,
            initial_epoch=len(self.history.history['loss']),
            callbacks=self.get_callbacks(),
            class_weight=class_weights,
            verbose=1
        )
        
        for key in self.history.history.keys():
            self.history.history[key].extend(history_finetune.history[key])
        
        print("✅ Training completed successfully!")
        return self.history
    
    def evaluate(self):
        """Evaluate on test set"""
        print("\nEvaluating model on test dataset...")
        
        test_gen = self.augmentor.get_test_generator()
        test_data = test_gen.flow_from_directory(
            os.path.join(self.config.DATA_PATH, 'test'),
            target_size=self.config.INPUT_SHAPE[:2],
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        results = self.model.evaluate(test_data, verbose=1)
        print(f"Test Accuracy: {results[1]*100:.2f}% | Precision: {results[2]*100:.2f}% | Recall: {results[3]*100:.2f}%")
        
        preds = self.model.predict(test_data, verbose=1)
        y_pred = np.argmax(preds, axis=1)
        y_true = test_data.classes
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.config.CROP_CLASSES))
        
        self._plot_confusion_matrix(y_true, y_pred)
        return results
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.config.CROP_CLASSES,
                    yticklabels=self.config.CROP_CLASSES)
        plt.title('Crop Classification Confusion Matrix (MobileNetV2)')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('stage1_confusion_matrix_mobilenetv2.png', dpi=300)
        print("Saved confusion matrix as 'stage1_confusion_matrix_mobilenetv2.png'")
    
    def save_model(self):
        """Save model and metadata"""
        self.model.save(self.config.MODEL_SAVE_PATH)
        print(f"\nModel saved at: {self.config.MODEL_SAVE_PATH}")
        
        metadata = {
            'model_name': 'Stage1_CropClassifier_MobileNetV2',
            'architecture': self.config.BASE_MODEL,
            'input_shape': self.config.INPUT_SHAPE,
            'num_classes': self.config.NUM_CLASSES,
            'crop_classes': self.config.CROP_CLASSES,
            'trained_on': datetime.now().isoformat()
        }
        
        meta_path = 'app/models/metadata/crop_metadata.json'
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {meta_path}")


def main():
    trainer = Stage1Trainer()
    trainer.build_model()
    train_data, val_data = trainer.prepare_data()
    trainer.train(train_data, val_data)
    trainer.evaluate()
    trainer.save_model()
    save_training_history(trainer.history, 'stage1_training_history_mobilenetv2.json')


if __name__ == '__main__':
    main()
