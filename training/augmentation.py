"""
Data augmentation utilities for robust model training
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from training.config import AugmentationConfig

class DataAugmentor:
    """Handles data augmentation for training"""
    
    def __init__(self, config=AugmentationConfig):
        self.config = config
        
    def get_train_generator(self):
        """Returns training data generator with augmentation"""
        return ImageDataGenerator(
            rescale=1./255,
            rotation_range=self.config.ROTATION_RANGE,
            width_shift_range=self.config.WIDTH_SHIFT_RANGE,
            height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
            shear_range=self.config.SHEAR_RANGE,
            zoom_range=self.config.ZOOM_RANGE,
            horizontal_flip=self.config.HORIZONTAL_FLIP,
            vertical_flip=self.config.VERTICAL_FLIP,
            brightness_range=self.config.BRIGHTNESS_RANGE,
            fill_mode=self.config.FILL_MODE
        )
    
    def get_val_generator(self):
        """Returns validation data generator (only rescaling)"""
        return ImageDataGenerator(rescale=1./255)
    
    def get_test_generator(self):
        """Returns test data generator (only rescaling)"""
        return ImageDataGenerator(rescale=1./255)
    
    @staticmethod
    def apply_custom_augmentation(image):
        """
        Apply custom augmentation for specific field conditions
        Simulates different lighting, weather, and camera angles
        """
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image
