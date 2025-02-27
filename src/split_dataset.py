import os
import random
import shutil

def split_dataset(
    images_dir="data/images/train",
    labels_dir="data/labels/train",
    masks_dir="data/masks/train",
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    """
    Splits dataset into train/val/test folders for images, labels, and (optionally) masks.
    Assumes images, labels, and masks share the same base filenames.
    """
    random.seed(seed)
    
    # Get list of all images in the original train folder
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    total_count = len(image_files)
    val_count = int(total_count * val_ratio)
    test_count = int(total_count * test_ratio)
    train_count = total_count - val_count - test_count

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    print(f"Total images: {total_count}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # Helper function to move files
    def move_files(file_list, split_name):
        # For images
        split_image_dir = f"data/images/{split_name}"
        os.makedirs(split_image_dir, exist_ok=True)

        # For labels
        split_label_dir = f"data/labels/{split_name}"
        os.makedirs(split_label_dir, exist_ok=True)

        # For masks (if used)
        split_mask_dir = f"data/masks/{split_name}"
        os.makedirs(split_mask_dir, exist_ok=True)

        for filename in file_list:
            # Move image
            src_img_path = os.path.join(images_dir, filename)
            dst_img_path = os.path.join(split_image_dir, filename)
            if os.path.exists(src_img_path):
                shutil.move(src_img_path, dst_img_path)

            # Move label
            label_filename = os.path.splitext(filename)[0] + ".txt"
            src_label_path = os.path.join(labels_dir, label_filename)
            dst_label_path = os.path.join(split_label_dir, label_filename)
            if os.path.exists(src_label_path):
                shutil.move(src_label_path, dst_label_path)

            # Move mask (if it exists)
            mask_filename_jpg = os.path.splitext(filename)[0] + ".jpg"
            mask_filename_png = os.path.splitext(filename)[0] + ".png"

            # We try .png first since that's common for masks
            src_mask_path = os.path.join(masks_dir, mask_filename_png)
            if os.path.exists(src_mask_path):
                dst_mask_path = os.path.join(split_mask_dir, mask_filename_png)
                shutil.move(src_mask_path, dst_mask_path)
            else:
                # If there's a chance your masks are .jpg
                src_mask_path = os.path.join(masks_dir, mask_filename_jpg)
                if os.path.exists(src_mask_path):
                    dst_mask_path = os.path.join(split_mask_dir, mask_filename_jpg)
                    shutil.move(src_mask_path, dst_mask_path)

    move_files(train_files, "train")
    move_files(val_files, "val")
    move_files(test_files, "test")

if __name__ == "__main__":
    split_dataset()
