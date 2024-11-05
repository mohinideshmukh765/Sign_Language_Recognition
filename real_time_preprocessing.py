import cv2
import numpy as np

def preprocess_frame(frame):
  
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to match the input shape of the CNN (e.g., 64x64)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    
    # Apply adaptive thresholding to highlight the hand/gesture
    thresholded_frame = cv2.adaptiveThreshold(resized_frame, 255, 
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)
    
    # Normalize pixel values to [0, 1] and add the channel dimension
    normalized_frame = thresholded_frame / 255.0
    preprocessed_frame = normalized_frame[..., np.newaxis]  # Add channel dimension for grayscale
    
    # Reshape for the model's expected input
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    
    return preprocessed_frame
