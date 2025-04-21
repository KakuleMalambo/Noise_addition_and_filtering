import os
import numpy as np
import matplotlib.pyplot as plt

# Import from our modules
from image_utils import load_images
from noise_generators import apply_noise_based_on_id
from filters import apply_mean_filter, apply_median_filter, apply_gaussian_filter
from visualization import display_images, display_filter_results
from evaluation import compare_filters, analyze_and_conclude

# Create a directory to save results if it doesn't exist
os.makedirs('results', exist_ok=True)

def main():
    
    # 1. Load Images and Add Noise
    print("\nTask 1: Loading images and adding noise...")
    original_images = load_images()
    noisy_images = apply_noise_based_on_id(original_images)
    display_images(original_images, noisy_images, "task1_")
    
    # 2. Apply Mean Filter
    print("\nTask 2: Applying mean filter...")
    display_filter_results(noisy_images, apply_mean_filter, [3, 5], filter_name="mean")
    
    # 3. Apply Median Filter
    print("\nTask 3: Applying median filter...")
    display_filter_results(noisy_images, apply_median_filter, [3, 5], filter_name="median")
    
    # 4. Apply Gaussian Filter
    print("\nTask 4: Applying Gaussian filter...")
    display_filter_results(noisy_images, apply_gaussian_filter, [3, 5], 
                          sigmas=[0.5, 1.0], filter_name="gaussian")
    
    # 5. Compare Filters and Conclude
    print("\nTask 5: Comparing filters and concluding...")
    compare_filters(original_images, noisy_images)
    analyze_and_conclude()
    
    print("\nAll tasks completed. Results saved in 'results' folder.")

if __name__ == "__main__":
    main()