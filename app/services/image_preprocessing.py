"""
Image preprocessing utilities for model inference
"""
import io
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handle image preprocessing for model inference"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def preprocess_for_stage1(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for Stage 1 (Crop Classification)
        
        Args:
            image_bytes: Raw image bytes from upload
            
        Returns:
            Preprocessed image array ready for model
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to target size
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to array
            img_array = np.array(image, dtype=np.float32)
            
            # EfficientNet preprocessing (already handled in model)
            # Just normalize to [0, 1]
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image for Stage 1: {str(e)}")
            raise
    
    def preprocess_for_stage2(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for Stage 2 (Disease Detection)
        Same as Stage 1 but can be customized per crop if needed
        
        Args:
            image_bytes: Raw image bytes from upload
            
        Returns:
            Preprocessed image array
        """
        return self.preprocess_for_stage1(image_bytes)
    
    def enhance_leaf_image(self, image_bytes: bytes) -> bytes:
        """
        Enhance leaf image for better detection
        - Increase contrast
        - Denoise
        - Sharpen edges
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Enhanced image bytes
        """
        try:
            # Load image with OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Denoise
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            # Sharpen
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Convert back to bytes
            _, buffer = cv2.imencode('.jpg', enhanced)
            return buffer.tobytes()
            
        except Exception as e:
            logger.warning(f"Error enhancing image: {str(e)}. Using original.")
            return image_bytes
    
    def validate_image(self, image_bytes: bytes, max_size: int = 10 * 1024 * 1024) -> Tuple[bool, str]:
        """
        Validate uploaded image
        
        Args:
            image_bytes: Image bytes to validate
            max_size: Maximum allowed size in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check size
        if len(image_bytes) > max_size:
            return False, f"Image too large. Maximum size is {max_size / (1024*1024):.1f}MB"
        
        # Try to open image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Check if it's a valid image
            image.verify()
            
            # Check dimensions
            width, height = image.size
            if width < 50 or height < 50:
                return False, "Image too small. Minimum size is 50x50 pixels"
            
            if width > 5000 or height > 5000:
                return False, "Image too large. Maximum dimension is 5000 pixels"
            
            return True, ""
            
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    def detect_leaf_in_image(self, image_bytes: bytes) -> Tuple[bool, float]:
        """
        Detect if image contains a leaf
        Uses color-based detection (green detection)
        
        Args:
            image_bytes: Image bytes
            
        Returns:
            Tuple of (contains_leaf, confidence)
        """
        try:
            # Load image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Define range for green color (leaves)
            lower_green = np.array([25, 40, 40])
            upper_green = np.array([95, 255, 255])
            
            # Create mask
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Calculate percentage of green pixels
            green_pixels = np.count_nonzero(mask)
            total_pixels = mask.size
            green_percentage = green_pixels / total_pixels
            
            # If more than 15% green, likely contains a leaf
            contains_leaf = green_percentage > 0.15
            confidence = min(green_percentage * 5, 1.0)  # Scale to 0-1
            
            return contains_leaf, confidence
            
        except Exception as e:
            logger.warning(f"Error detecting leaf: {str(e)}")
            return True, 0.5  # Default to true with low confidence


# Global preprocessor instance
preprocessor = ImagePreprocessor()
