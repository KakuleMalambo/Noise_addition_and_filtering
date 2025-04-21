import cv2

def apply_mean_filter(image, window_size):
    """Apply mean filter to image"""
    return cv2.blur(image, (window_size, window_size))

def apply_median_filter(image, window_size):
    """Apply median filter to image
    Note: window_size must be odd in OpenCV's medianBlur
    """
    # Ensure odd size for medianBlur
    if window_size % 2 == 0:
        window_size += 1
    return cv2.medianBlur(image, window_size)

def apply_gaussian_filter(image, window_size, sigma=0):
    """Apply Gaussian filter to image
    Note: window_size must be odd in OpenCV's GaussianBlur
    """
    # Ensure odd size for GaussianBlur
    if window_size % 2 == 0:
        window_size += 1
    return cv2.GaussianBlur(image, (window_size, window_size), sigma)
