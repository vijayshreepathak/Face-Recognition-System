import cv2
import numpy as np
import pytesseract
from typing import Tuple, List, Dict
import re

class ImageProcessor:
    def __init__(self):
        self.medicine_info = {
            "name": "",
            "dosage": "",
            "instructions": "",
            "warnings": "",
            "expiry_date": ""
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(morph)
        
        return denoised
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions of text in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply MSER (Maximally Stable Extremal Regions)
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        # Convert regions to bounding boxes
        boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            boxes.append((x, y, w, h))
        
        return boxes
    
    def extract_medicine_info(self, text: str) -> Dict[str, str]:
        """Extract structured information from OCR text"""
        # Regular expressions for different types of information
        patterns = {
            "name": r"(?i)(?:name|medication|drug):?\s*([^\n]+)",
            "dosage": r"(?i)(?:dosage|strength):?\s*([^\n]+)",
            "instructions": r"(?i)(?:take|use|directions):?\s*([^\n]+)",
            "warnings": r"(?i)(?:warning|caution|precaution):?\s*([^\n]+)",
            "expiry_date": r"(?i)(?:expiry|expiration):?\s*([^\n]+)"
        }
        
        info = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                info[key] = match.group(1).strip()
            else:
                info[key] = ""
        
        return info
    
    def process_image(self, image: np.ndarray) -> Dict[str, str]:
        """Process the image and extract medicine information"""
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Extract text using OCR
        text = pytesseract.image_to_string(processed_image)
        
        # Extract structured information
        medicine_info = self.extract_medicine_info(text)
        
        return medicine_info 