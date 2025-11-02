"""
Training Script for Stage 1: Crop Classification (MobileNetV2, CPU-optimized)
Wrapper that calls the main training module

Replaces EfficientNetB0 with MobileNetV2 and adds:
 - class weighting to handle imbalance
 - CPU-friendly architecture and smaller head
 - partial fine-tuning of base model in phase 2
 - keeps same pipeline, callbacks, evaluation, and saving logic
"""

import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Reduce TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from training.config import Stage1Config
from training.augmentation import DataAugmentor
from training.utils import create_directories, save_training_history, plot_training_history

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: enable mixed precision if your CPU benefits from it (most CPUs won't)
# try:
#     from tensorflow.keras import mixed_precision
#     mixed_precision.set_global_policy("mixed_float16")
# except Exception:
#     pass


class Stage1Trainer:
    """Stage 1: Crop Classification Trainer (MobileNetV2)"""

    def __init__(self):
        self.config = Stage1Config()
        self.augmentor = DataAugmentor()
        self.model = None
        self.history = None
        self.class_weights = None

        # Set random seeds
        np.random.seed(self.config.RANDOM_SEED)
        tf.random.set_seed(self.config.RANDOM_SEED)

        # Create directories used by training pipeline
        create_directories([
            self.config.CHECKPOINT_PATH,
            self.config.MODEL_DIR,
            self.config.LOG_DIR
        ])

        # Log header
        print("\n" + "=" * 70)
        print(" " * 15 + "STAGE 1: CROP CLASSIFICATION TRAINING (MobileNetV2)")
        print("=" * 70)

    def build_model(self):
        """Build MobileNetV2-based model for crop classification"""
        print("\n[1/6] Building Model Architecture (MobileNetV2)...")
        # Use config values but be more CPU friendly in head size
        input_shape = tuple(self.config.INPUT_SHAPE)
        num_classes = self.config.NUM_CLASSES

        print(f"Base Model: MobileNetV2")
        print(f"Input Shape: {input_shape}")
        print(f"Number of Crops: {num_classes}")

        # Load pre-trained MobileNetV2 as base
        # give it a deterministic name so we can find it later for unfreezing
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            name='mobilenetv2_base'
        )

        # Freeze base model initially
        base_model.trainable = False

        # Build classification head (smaller head for CPU speed)
        inputs = keras.Input(shape=input_shape)
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)

        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(self.config.DROPOUT_RATE if hasattr(self.config, "DROPOUT_RATE") else 0.3,
                           name='dropout_1')(x)

        # a modest dense head tuned for CPU
        x = layers.Dense(384, activation='relu', name='fc_1')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(self.config.DROPOUT_RATE * 0.6 if hasattr(self.config, "DROPOUT_RATE") else 0.25,
                           name='dropout_2')(x)

        x = layers.Dense(192, activation='relu', name='fc_2')(x)
        x = layers.Dropout((self.config.DROPOUT_RATE * 0.5) if hasattr(self.config, "DROPOUT_RATE") else 0.2,
                           name='dropout_3')(x)

        outputs = layers.Dense(num_classes, activation='softmax', name='crop_output')(x)

        self.model = keras.Model(inputs, outputs, name='CropClassifier_MobileNetV2')

        # Compile model: keep same metrics as before
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )

        print(f"✓ Model built successfully!")
        print(f"  Total parameters: {self.model.count_params():,}")

        # compute trainable parameters count
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        print(f"  Trainable parameters: {trainable_params:,}")

        return self.model

    def prepare_data(self):
        """Prepare data generators and compute class weights"""
        print("\n[2/6] Preparing Data Generators...")

        data_path = self.config.DATA_PATH

        if not data_path.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_path}\n"
                f"Please run: python scripts/02_prepare_data.py"
            )

        # Use DataAugmentor from training.augmentation to keep behavior consistent
        train_gen = self.augmentor.get_train_generator()
        val_gen = self.augmentor.get_val_generator()

        # Create directory iterators
        train_data = train_gen.flow_from_directory(
            str(data_path / 'train'),
            target_size=self.config.INPUT_SHAPE[:2],
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=self.config.RANDOM_SEED
        )

        val_data = val_gen.flow_from_directory(
            str(data_path / 'validation'),
            target_size=self.config.INPUT_SHAPE[:2],
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        # Update crop classes and num_classes based on detected folders
        self.config.CROP_CLASSES = list(train_data.class_indices.keys())
        self.config.NUM_CLASSES = len(self.config.CROP_CLASSES)

        # Compute class weights to help with imbalance
        try:
            num_classes = self.config.NUM_CLASSES
            # y is array of class indices used by the generator
            y = np.array(train_data.classes)
            classes = np.arange(num_classes)
            weights_array = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=classes,
                                                              y=y)
            # map indices 0..N-1 to weights
            self.class_weights = {int(i): float(w) for i, w in enumerate(weights_array)}
        except Exception as e:
            print(f"⚠ Could not compute class weights automatically: {e}")
            self.class_weights = None

        print(f"✓ Data loaded successfully!")
        print(f"  Training samples: {train_data.samples}")
        print(f"  Validation samples: {val_data.samples}")
        print(f"  Crop classes: {', '.join(self.config.CROP_CLASSES)}")
        if self.class_weights:
            print(f"  Class weights computed for imbalance handling")

        return train_data, val_data

    def get_callbacks(self):
        """Setup training callbacks"""
        print("\n[3/6] Setting up Callbacks...")

        callbacks = [
            ModelCheckpoint(
                filepath=str(self.config.CHECKPOINT_PATH / 'best_model.h5'),
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
                log_dir=str(self.config.LOG_DIR / 'stage1' / datetime.now().strftime("%Y%m%d-%H%M%S")),
                histogram_freq=1
            )
        ]

        print(f"✓ Callbacks configured!")
        return callbacks

    def train(self, train_data, val_data):
        """Train the model (phase 1: frozen base, phase 2: partial/unfreeze)"""
        print("\n[4/6] Starting Training...")
        print(f"  Epochs: {self.config.EPOCHS}")
        print(f"  Batch size: {self.config.BATCH_SIZE}")
        print(f"  Learning rate: {self.config.LEARNING_RATE}")

        callbacks = self.get_callbacks()

        # Phase 1: Train with frozen base model
        print("\n--- Phase 1: Training with frozen base model ---")
        history1 = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.EPOCHS // 2,
            callbacks=callbacks,
            class_weight=self.class_weights if self.class_weights else None,
            verbose=1
        )

        # Phase 2: Fine-tune - unfreeze top layers of base model
        print("\n--- Phase 2: Fine-tuning top layers of base model ---")

        # Find base model by name and unfreeze the last N layers for fine-tuning
        try:
            base = self.model.get_layer('mobilenetv2_base')
        except Exception:
            # fallback: assume layer index 2 (original behavior)
            base = self.model.layers[2]

        # Unfreeze top layers only (keep many layers frozen for safety on CPU)
        # We'll unfreeze the last `FINE_TUNE_AT` layers of the base model
        FINE_TUNE_AT = getattr(self.config, 'FINE_TUNE_AT', 50)  # default to unfreeze last 50 layers
        try:
            total_layers = len(base.layers)
            for i, layer in enumerate(base.layers):
                layer.trainable = (i >= (total_layers - FINE_TUNE_AT))
            print(f"  ✓ Unfroze last {FINE_TUNE_AT} layers out of {total_layers} base layers for fine-tuning")
        except Exception:
            base.trainable = True
            print("  ⚠ could not fine-tune individual base layers; base model fully unfrozen for fine-tuning")

        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE / 10),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )

        # Continue training (with initial_epoch set)
        history2 = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.EPOCHS,
            initial_epoch=len(history1.history['loss']),
            callbacks=callbacks,
            class_weight=self.class_weights if self.class_weights else None,
            verbose=1
        )

        # Combine histories (preserve phase1 then phase2)
        self.history = history1
        for key in history1.history.keys():
            self.history.history[key].extend(history2.history.get(key, []))

        print("\n✓ Training completed!")

        return self.history

    def evaluate(self):
        """Evaluate on test set"""
        print("\n[5/6] Evaluating Model...")

        test_gen = self.augmentor.get_test_generator()
        test_data = test_gen.flow_from_directory(
            str(self.config.DATA_PATH / 'test'),
            target_size=self.config.INPUT_SHAPE[:2],
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        # Evaluate
        results = self.model.evaluate(test_data, verbose=1)

        print("\n" + "=" * 70)
        print("TEST SET RESULTS:")
        print("=" * 70)
        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1] * 100:.2f}%")
        print(f"Test Precision: {results[2] * 100:.2f}%")
        print(f"Test Recall: {results[3] * 100:.2f}%")
        print("=" * 70)

        # Get predictions
        predictions = self.model.predict(test_data, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_data.classes

        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            y_true,
            y_pred,
            target_names=self.config.CROP_CLASSES,
            digits=4
        ))

        # Plot confusion matrix
        self._plot_confusion_matrix(y_true, y_pred)

        return results

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.config.CROP_CLASSES,
            yticklabels=self.config.CROP_CLASSES,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Crop Classification - Confusion Matrix', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        save_path = self.config.LOG_DIR / 'stage1_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved: {save_path}")
        plt.close()

    def save_model(self):
        """Save model and metadata"""
        print("\n[6/6] Saving Model...")

        # Save model (use same path from config)
        self.model.save(str(self.config.MODEL_SAVE_PATH))
        print(f"✓ Model saved: {self.config.MODEL_SAVE_PATH}")

        # Save metadata
        metadata = {
            'model_name': 'Stage1_CropClassifier_MobileNetV2',
            'architecture': 'MobileNetV2',
            'input_shape': list(self.config.INPUT_SHAPE),
            'num_classes': self.config.NUM_CLASSES,
            'crop_classes': self.config.CROP_CLASSES,
            'training_date': datetime.now().isoformat(),
            'total_params': int(self.model.count_params()),
            'config': {
                'batch_size': self.config.BATCH_SIZE,
                'epochs': self.config.EPOCHS,
                'learning_rate': self.config.LEARNING_RATE,
                'dropout_rate': getattr(self.config, 'DROPOUT_RATE', None)
            }
        }

        metadata_path = self.config.MODEL_DIR / 'metadata' / 'crop_metadata.json'
        metadata_path.parent.mkdir(exist_ok=True)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved: {metadata_path}")

        # Save training history
        history_path = self.config.LOG_DIR / 'stage1_training_history.json'
        save_training_history(self.history, str(history_path))
        print(f"✓ Training history saved: {history_path}")

        # Plot training curves
        plot_path = self.config.LOG_DIR / 'stage1_training_curves.png'
        plot_training_history(self.history, str(plot_path))
        print(f"✓ Training curves saved: {plot_path}")

    def run_full_pipeline(self):
        """Execute complete training pipeline"""
        try:
            # Build model
            self.build_model()

            # Prepare data
            train_data, val_data = self.prepare_data()

            # Train
            self.train(train_data, val_data)

            # Evaluate
            self.evaluate()

            # Save
            self.save_model()

            print("\n" + "=" * 70)
            print(" " * 20 + "TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print("\nNext step: Run python scripts/04_train_stage2.py")
            print("=" * 70)

        except Exception as e:
            print(f"\n✗ Error during training: {str(e)}")
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main function"""
    # Check GPU availability (keep original behavior)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("⚠ No GPU found. Training will use CPU (slower)")

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")

    # Initialize and run trainer
    trainer = Stage1Trainer()
    trainer.run_full_pipeline()


if __name__ == '__main__':
    main()
