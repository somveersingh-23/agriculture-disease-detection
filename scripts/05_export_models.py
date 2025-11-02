"""
Export Models for Deployment
Converts models to TFLite for mobile deployment (optional)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
import json


class ModelExporter:
    """Export trained models"""
    
    def __init__(self):
        self.model_dir = Path('app/models')
        self.export_dir = Path('app/models/exported')
        self.export_dir.mkdir(exist_ok=True)
    
    def export_to_tflite(self, model_path, output_name):
        """Convert model to TensorFlow Lite"""
        print(f"\nExporting {output_name}...")
        
        # Load model
        model = tf.keras.models.load_model(str(model_path))
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save
        tflite_path = self.export_dir / f'{output_name}.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get size
        size_mb = tflite_path.stat().st_size / (1024 * 1024)
        print(f"✓ Exported: {tflite_path} ({size_mb:.2f} MB)")
    
    def export_all(self):
        """Export all models"""
        print("="*60)
        print("EXPORTING MODELS FOR DEPLOYMENT")
        print("="*60)
        
        # Export Stage 1
        stage1_path = self.model_dir / 'stage1_crop_classifier.h5'
        if stage1_path.exists():
            self.export_to_tflite(stage1_path, 'stage1_crop_classifier')
        
        # Export Stage 2
        stage2_dir = self.model_dir / 'stage2_disease_models'
        if stage2_dir.exists():
            for model_file in stage2_dir.glob('*.h5'):
                crop_name = model_file.stem.replace('_disease', '')
                self.export_to_tflite(model_file, f'stage2_{crop_name}')
        
        print("\n✓ All models exported!")


if __name__ == '__main__':
    exporter = ModelExporter()
    exporter.export_all()
