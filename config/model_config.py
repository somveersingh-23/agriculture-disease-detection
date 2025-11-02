"""
Model Configuration
"""

class ModelConfig:
    """Model architecture configuration"""
    
    # Available base models
    AVAILABLE_MODELS = {
        'EfficientNetB0': {
            'input_size': (224, 224),
            'params': '5.3M',
            'accuracy': '97.1%',
            'speed': 'Fast'
        },
        'EfficientNetB1': {
            'input_size': (240, 240),
            'params': '7.8M',
            'accuracy': '97.5%',
            'speed': 'Medium'
        },
        'MobileNetV2': {
            'input_size': (224, 224),
            'params': '3.5M',
            'accuracy': '95.8%',
            'speed': 'Very Fast'
        },
        'ResNet50': {
            'input_size': (224, 224),
            'params': '25.6M',
            'accuracy': '96.2%',
            'speed': 'Slow'
        }
    }
    
    # Selected model for production
    SELECTED_MODEL = 'EfficientNetB0'
    
    # Training hyperparameters
    HYPERPARAMETERS = {
        'initial_lr': 0.001,
        'final_lr': 1e-7,
        'batch_size': 32,
        'epochs': 50,
        'patience': 10
    }
