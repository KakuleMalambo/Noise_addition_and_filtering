import cv2
import numpy as np

def load_images():
    """Load images or create sample images if not available"""
    # You can replace these paths with your actual image paths
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    images = []
    
    # Try to load images, if not available, use sample images
    for i, path in enumerate(image_paths):
        try:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except (FileNotFoundError, Exception):
            # Create sample images if files not found
            print(f"Image {path} not found, creating sample image")
            img = create_sample_image(i)
        
        # Resize for consistency
        img = cv2.resize(img, (512, 512))
        images.append(img)
    
    return images

def create_sample_image(index):
    """Create a sample image if no image is provided"""
    if index == 0:
        # Create a gradient image
        x = np.linspace(0, 1, 512)
        y = np.linspace(0, 1, 512)
        xx, yy = np.meshgrid(x, y)
        img = np.stack([xx, yy, np.zeros_like(xx)], axis=2)
        return (img * 255).astype(np.uint8)
    elif index == 1:
        # Create a checkerboard pattern
        x = np.linspace(0, 7, 512)
        y = np.linspace(0, 7, 512)
        xx, yy = np.meshgrid(x, y)
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        pattern = (np.floor(xx) + np.floor(yy)) % 2
        img[:,:,0] = pattern * 255
        img[:,:,1] = pattern * 255
        img[:,:,2] = pattern * 255
        return img
    else:
        # Create a circle pattern
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(img, (256, 256), 200, (255, 0, 0), -1)
        cv2.circle(img, (256, 256), 150, (0, 255, 0), -1)
        cv2.circle(img, (256, 256), 100, (0, 0, 255), -1)
        return img
