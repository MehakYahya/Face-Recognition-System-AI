import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Constants
DATASET_PATH = Path(r"C:\Users\syeda\Downloads\p1\Student attendance.v1i.retinanet")
IMG_SIZE = 128

def preprocess_data(dataset_path=DATASET_PATH, img_size=IMG_SIZE):
    """
    Preprocesses images from a dataset into train, validation, and test splits.

    Parameters:
        dataset_path (Path or str): Path to the dataset directory containing subfolders 'train', 'valid', 'test'.
        img_size (int): Target size for resizing images.

    Returns:
        dict: A dictionary with keys ['train', 'valid', 'test'], each containing:
            - "images": NumPy array of preprocessed images.
            - "labels": NumPy array of corresponding labels.
    """
    data_splits = ["train", "valid", "test"]
    data_dict = {}
    label_map = {}  # For handling multiple classes dynamically
    label_counter = 0

    # Check if the dataset directory exists
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    print(f"Dataset directory: {dataset_path}")
    print(f"Available splits: {data_splits}")

    for split in data_splits:
        img_list = []
        label_list = []  # Labels list for the split
        split_path = Path(dataset_path) / split

        if not split_path.exists():
            print(f"Warning: {split.capitalize()} split path not found: {split_path}")
            continue

        print(f"Processing {split.capitalize()} split...")
        for file in split_path.glob("*"):
            if file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    img = cv2.imread(str(file))
                    if img is not None:
                        # Resize and normalize the image
                        img_resized = cv2.resize(img, (img_size, img_size)) / 255.0
                        img_list.append(img_resized)

                        # Create label mapping dynamically
                        if file.name not in label_map:
                            label_map[file.name] = label_counter
                            label_counter += 1
                        label_list.append(label_map[file.name])
                    else:
                        print(f"Warning: Could not read image file: {file}")
                except Exception as e:
                    print(f"Error loading image {file}: {e}")

        if img_list:
            # Convert to NumPy arrays
            data_dict[split] = {
                "images": np.array(img_list, dtype="float32"),
                "labels": np.array(label_list, dtype="int32"),
            }
            print(f"Loaded {len(img_list)} images for {split.capitalize()} split.")
        else:
            print(f"No valid images found in {split.capitalize()} split.")

    return data_dict, label_map

if __name__ == "__main__":
    try:
        data_splits, label_map = preprocess_data()
        for split, data in data_splits.items():
            print(f"{split.capitalize()} size: {len(data['images'])} images, {len(data['labels'])} labels")
    except Exception as e:
        print(f"Error: {e}")
