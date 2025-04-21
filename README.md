# Image Denoising and Filter Comparison

This project demonstrates the application and comparison of various noise reduction filters (mean, median, Gaussian) on images corrupted by different types of noise (JPEG compression, salt & pepper, impulse). It includes visualization and quantitative evaluation (PSNR, SSIM) of filter performance.

## Project Structure

- `main.py` — Main script to run all tasks.
- `image_utils.py` — Image loading and sample image generation.
- `noise_generators.py` — Functions to add different types of noise.
- `filters.py` — Implementations of mean, median, and Gaussian filters.
- `visualization.py` — Visualization utilities for images and filter results.
- `evaluation.py` — PSNR/SSIM calculation, filter comparison, and analysis.
- `results/` — Output directory for generated images and plots.

## Requirements

- Python 3.x
- numpy
- opencv-python
- matplotlib
- scikit-image

Install dependencies with:

```bash
pip install numpy opencv-python matplotlib scikit-image
```

## Usage

1. Place your images as `image1.jpg`, `image2.jpg`, `image3.jpg` in the project directory, or let the script generate sample images.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Results and visualizations will be saved in the `results/` folder.

## Notes

- If images are not found, synthetic images will be generated.
- The script prints filter comparison tables and analysis to the console.
