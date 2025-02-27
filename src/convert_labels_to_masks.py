import os
import cv2
import numpy as np

# Directories for images, labels, and output masks
images_dir = "data/images/train"
labels_dir = "data/labels/train"
masks_dir = "data/masks/train"  # Output folder for masks

# Create the output directory if it doesn't exist
os.makedirs(masks_dir, exist_ok=True)

# Process each image in the images directory
for filename in os.listdir(images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(images_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width, _ = image.shape
        
        # Create a blank mask (all zeros = black background)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Determine the corresponding label file (assumes same base name with .txt extension)
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")
        
        # Read the label file if it exists
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    # YOLO format: class_id, x_center, y_center, width, height (all normalized)
                    class_id, x_center, y_center, w, h = map(float, parts)
                    
                    # Convert normalized coordinates to absolute pixel values
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)
                    
                    # Draw a filled white rectangle on the mask for the bounding box
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        
        # Save the mask as a PNG image
        mask_output_path = os.path.join(masks_dir, base_name + ".png")
        cv2.imwrite(mask_output_path, mask)

print("Conversion complete. Masks saved to:", masks_dir)
