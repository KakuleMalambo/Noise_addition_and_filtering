import matplotlib.pyplot as plt
import numpy as np

def display_images(original_images, noisy_images, title_prefix=""):
    """Display original and noisy images side by side"""
    fig, axes = plt.subplots(len(original_images), 2, figsize=(12, 6 * len(original_images)))
    
    # Update noise types order to match the noise application
    noise_types = ["Compression (JPEG)", "Salt and pepper", "Impulse"]
    
    for i in range(len(original_images)):
        if len(original_images) == 1:
            ax1, ax2 = axes
        else:
            ax1, ax2 = axes[i]
        
        ax1.imshow(original_images[i])
        ax1.set_title(f"Original Image {i+1}")
        ax1.axis('off')
        
        ax2.imshow(noisy_images[i])
        ax2.set_title(f"Noisy Image {i+1} ({noise_types[i]})")
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'results/{title_prefix}original_vs_noisy.png')
    plt.show()

def display_filter_results(noisy_images, filter_function, window_sizes, sigmas=None, filter_name=""):
    """Display results of filtering with different parameters"""
    # Update noise types order to match the noise application
    noise_types = ["Compression (JPEG)", "Salt and pepper", "Impulse"]
    
    # For Mean and Median filters
    if sigmas is None:
        for i, img in enumerate(noisy_images):
            fig, axes = plt.subplots(1, len(window_sizes) + 1, figsize=(15, 5))
            
            axes[0].imshow(img)
            axes[0].set_title(f"Noisy Image ({noise_types[i]})")
            axes[0].axis('off')
            
            for j, size in enumerate(window_sizes):
                if filter_name == "median":
                    # Ensure odd size for median
                    if size % 2 == 0:
                        size += 1
                
                filtered_img = filter_function(img, size)
                axes[j+1].imshow(filtered_img)
                axes[j+1].set_title(f"{filter_name.capitalize()} Filter ({size}x{size})")
                axes[j+1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'results/{filter_name}_filter_image{i+1}.png')
            plt.show()
    
    # For Gaussian filter with multiple sigma values
    else:
        for i, img in enumerate(noisy_images):
            fig, axes = plt.subplots(len(sigmas), len(window_sizes) + 1, figsize=(15, 5 * len(sigmas)))
            
            for s in range(len(sigmas)):
                axes[s, 0].imshow(img)
                axes[s, 0].set_title(f"Noisy Image ({noise_types[i]})")
                axes[s, 0].axis('off')
                
                for w in range(len(window_sizes)):
                    sigma = sigmas[s]
                    window_size = window_sizes[w]
                    
                    # Ensure window size is odd for Gaussian blur
                    if window_size % 2 == 0:
                        window_size += 1
                        
                    filtered_img = filter_function(img, window_size, sigma)
                    axes[s, w+1].imshow(filtered_img)
                    axes[s, w+1].set_title(f"Gaussian (size={window_size}, σ={sigma})")
                    axes[s, w+1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'results/gaussian_filter_image{i+1}.png')
            plt.show()

def create_comparison_visualizations(original_images, noisy_images, best_filters):
    """Create visual comparison of best filters for each noise type"""
    # Update noise types order to match the noise application
    noise_types = ["Compression (JPEG)", "Salt and pepper", "Impulse"]
    
    from filters import apply_mean_filter, apply_median_filter, apply_gaussian_filter
    from evaluation import calculate_psnr
    
    for i, (orig_img, noisy_img) in enumerate(zip(original_images, noisy_images)):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(orig_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Noisy image
        axes[1].imshow(noisy_img)
        axes[1].set_title(f"Noisy Image\n({noise_types[i]})")
        axes[1].axis('off')
        
        # Mean filter (3x3)
        mean_filtered = apply_mean_filter(noisy_img, 3)
        axes[2].imshow(mean_filtered)
        axes[2].set_title(f"Mean Filter (3x3)\nPSNR: {calculate_psnr(orig_img, mean_filtered):.2f}")
        axes[2].axis('off')
        
        # Best filter for this noise type
        best_filter = best_filters[i]
        if best_filter["type"] == "Mean":
            size = best_filter["params"]["window_size"]
            best_filtered = apply_mean_filter(noisy_img, size)
            title = f"Mean Filter ({size}x{size})"
        elif best_filter["type"] == "Median":
            size = best_filter["params"]["window_size"]
            best_filtered = apply_median_filter(noisy_img, size)
            title = f"Median Filter ({size}x{size})"
        else:  # Gaussian
            size = best_filter["params"]["window_size"]
            sigma = best_filter["params"]["sigma"]
            best_filtered = apply_gaussian_filter(noisy_img, size, sigma)
            title = f"Gaussian Filter ({size}x{size}, σ={sigma})"
        
        axes[3].imshow(best_filtered)
        axes[3].set_title(f"{title}\nPSNR: {calculate_psnr(orig_img, best_filtered):.2f}")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/comparison_{noise_types[i].replace(" ", "_").replace("(", "").replace(")", "")}.png')
        plt.show()
