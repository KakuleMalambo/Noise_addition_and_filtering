import cv2
import numpy as np
from skimage.util import random_noise

def add_compression_noise(image):
    """Add JPEG compression noise to the image"""
    # Define the compression quality (lower = more compression)
    compression_quality = 5
    
    # Encode and then decode the image with JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    compressed_img = cv2.imdecode(encimg, 1)
    
    # Convert back to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        compressed_img = cv2.cvtColor(compressed_img, cv2.COLOR_BGR2RGB)
    
    return compressed_img

def add_impulse_noise(image):
    """Add impulse noise to the image"""
    # Define the probability of noise (higher = more noise)
    prob = 0.05
    
    # Create a copy of the image
    noisy_image = image.copy()
    
    # Generate random positions
    height, width = image.shape[:2]
    num_pixels = int(height * width * prob)
    
    # Add random bright and dark pixels
    for _ in range(num_pixels):
        # Random position
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        
        # Set value - either black (0) or white (255)
        noise_val = 255 if np.random.random() < 0.5 else 0
        
        # Apply noise
        if len(image.shape) == 3:  # Color image
            noisy_image[y, x, :] = noise_val
        else:  # Grayscale image
            noisy_image[y, x] = noise_val
    
    return noisy_image

def add_salt_and_pepper_noise(image):
    """Add salt and pepper noise to the image"""
    # Use skimage's random_noise function to add salt and pepper noise
    noisy_image = random_noise(image, mode='s&p', amount=0.05)
    return (noisy_image * 255).astype(np.uint8)

def apply_noise_based_on_id(images):
    """Apply noise to images based on student ID"""
    # Noise 1: Compression (JPEG)
    # Noise 2: Salt and pepper 
    # Noise 3: Impulse
    
    noisy_images = []
    for i, img in enumerate(images):
        if i == 0:
            noisy_images.append(add_compression_noise(img))
        elif i == 1:
            noisy_images.append(add_salt_and_pepper_noise(img))
        elif i == 2:
            noisy_images.append(add_impulse_noise(img))
    
    return noisy_images
