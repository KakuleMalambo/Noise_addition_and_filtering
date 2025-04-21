import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from filters import apply_mean_filter, apply_median_filter, apply_gaussian_filter
from visualization import create_comparison_visualizations

def calculate_psnr(original, filtered):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original.astype(np.float64) - filtered.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, filtered):
    """Calculate Structural Similarity Index"""
    if len(original.shape) == 3:  # Color image
        # Convert to grayscale for SSIM
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
        return ssim(original_gray, filtered_gray)
    else:  # Already grayscale
        return ssim(original, filtered)

def compare_filters(original_images, noisy_images):
    """Compare all filters and their parameters"""
    # Define parameters
    mean_sizes = [3, 5]
    median_sizes = [3, 5]
    gaussian_sizes = [3, 5]
    gaussian_sigmas = [0.5, 1.0]
    
    # Update noise types order to match the noise application
    noise_types = ["Compression (JPEG)", "Salt and pepper", "Impulse"]
    
    # Prepare table headers
    print("\nFilter Comparison Results:\n")
    print(f"{'Noise Type':<20} {'Filter':<20} {'Parameters':<20} {'PSNR':<10} {'SSIM':<10}")
    print("-" * 80)
    
    # Compare filters for each image
    for i, (orig_img, noisy_img) in enumerate(zip(original_images, noisy_images)):
        noise_type = noise_types[i]
        
        # Mean Filter
        for size in mean_sizes:
            filtered = apply_mean_filter(noisy_img, size)
            psnr = calculate_psnr(orig_img, filtered)
            ssim_val = calculate_ssim(orig_img, filtered)
            print(f"{noise_type:<20} {'Mean':<20} {f'Size={size}x{size}':<20} {psnr:<10.2f} {ssim_val:<10.4f}")
        
        # Median Filter
        for size in median_sizes:
            # Ensure odd size for medianBlur
            if size % 2 == 0:
                size += 1
            filtered = apply_median_filter(noisy_img, size)
            psnr = calculate_psnr(orig_img, filtered)
            ssim_val = calculate_ssim(orig_img, filtered)
            print(f"{noise_type:<20} {'Median':<20} {f'Size={size}x{size}':<20} {psnr:<10.2f} {ssim_val:<10.4f}")
        
        # Gaussian Filter
        for size in gaussian_sizes:
            # Ensure odd size for GaussianBlur
            if size % 2 == 0:
                size += 1
            for sigma in gaussian_sigmas:
                filtered = apply_gaussian_filter(noisy_img, size, sigma)
                psnr = calculate_psnr(orig_img, filtered)
                ssim_val = calculate_ssim(orig_img, filtered)
                print(f"{noise_type:<20} {'Gaussian':<20} {f'Size={size}x{size}, σ={sigma}':<20} {psnr:<10.2f} {ssim_val:<10.4f}")
    
    # Define the best filter for each noise type - updated order
    best_filters = [
        # For JPEG Compression Noise
        {"type": "Gaussian", "params": {"window_size": 5, "sigma": 0.5}},
        # For Salt and Pepper Noise (now second)
        {"type": "Median", "params": {"window_size": 3}},
        # For Impulse Noise (now third)
        {"type": "Median", "params": {"window_size": 3}}
    ]
    
    # Create visual table for report
    create_comparison_visualizations(original_images, noisy_images, best_filters)

def analyze_and_conclude():
    """Analyze and provide conclusions about filter effectiveness"""
    print("\n\nAnalysis and Conclusions:")
    print("-" * 50)
    
    print("\n1. Analysis by Noise Type:")
    
    print("\n   a. Compression (JPEG) Noise:")
    print("      - Impact: Adds blocky artifacts and can cause loss of fine details")
    print("      - Best Filter: Gaussian Filter")
    print("      - Reason: Gaussian smoothing helps reduce the blockiness while preserving edges better than mean filtering")
    
    print("\n   b. Impulse Noise:")
    print("      - Impact: Adds random extreme value pixels (black or white)")
    print("      - Best Filter: Median Filter")
    print("      - Reason: Median filtering is specifically designed to remove outliers without affecting the rest of the image")
    
    print("\n   c. Salt and Pepper Noise:")
    print("      - Impact: Adds scattered white and black pixels")
    print("      - Best Filter: Median Filter")
    print("      - Reason: Median filtering effectively removes isolated noise pixels without blurring edges")
    
    print("\n2. Parameter Impact:")
    print("\n   a. Window Size:")
    print("      - Larger windows (5x5) provide more smoothing but can blur details")
    print("      - Smaller windows (3x3) preserve more details but may not fully remove noise")
    print("      - For impulse and salt & pepper noise, 3x3 median filter is often sufficient")
    print("      - For compression noise, larger windows may be needed to smooth out larger artifacts")
    
    print("\n   b. Sigma (for Gaussian Filter):")
    print("      - Lower sigma values (0.5) preserve more details")
    print("      - Higher sigma values (1.0+) provide more smoothing but increase blurring")
    print("      - For compression noise, a moderate sigma value balances artifact removal and detail preservation")
    
    print("\n3. General Conclusions:")
    print("\n   a. Filter Selection Guidelines:")
    print("      - Salt and Pepper/Impulse noise: Median filter is almost always the best choice")
    print("      - JPEG Compression noise: Gaussian filter with carefully tuned parameters works best")
    print("      - Mean filter is rarely the optimal choice but is computationally efficient")
    
    print("\n   b. Real-world Applications:")
    print("      - Digital photography: Combine median and Gaussian filters for mixed noise")
    print("      - Medical imaging: Use median filters for impulse noise and adaptive filters for complex noise")
    print("      - Document scanning: Use specialized filters to enhance text while reducing compression artifacts")
