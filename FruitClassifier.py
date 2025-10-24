# Fruit Classifier using Machine Learning
from email.mime import image
import pandas as pd
import os
from pathlib import Path
import sys
import cv2
from skimage.feature import local_binary_pattern


#load images
def load_data(folder_paths):
    all_image_paths = []
    all_labels = []

    print("Starting data loading process...")

    if not folder_paths:
        print("No folder paths provided.")
        return pd.DataFrame(columns=['image_path', 'label'])
    
    try:
        first_path = folder_paths[next(iter(folder_paths))][0]
        if not Path(first_path).parent.exists():
            print(f"Error: The directory {Path(first_path).parent} does not exist.", file=sys.stderr)
            return pd.DataFrame(columns=['image_path', 'label'])
    except Exception as e:
        print(f"Error accessing the directory: {e}", file=sys.stderr)
        return pd.DataFrame(columns=['image_path', 'label'])

    for label, paths in folder_paths.items():
        print("Loading data for label:", label)
        label_image_count = 0
        for folder_path in paths:
            path = Path(folder_path)
            if not path.exists():
                print(f"Warning: The directory {path} does not exist. Skipping...", file=sys.stderr)
                continue
            #get all files in the folder (none in subfolders)
            image_files = list(path.glob("*.jpg"))
            
            if not image_files:
                print(f"Warning: No image files found in {path}.", file=sys.stderr)
                continue

            for image_file in image_files:
                all_image_paths.append(str(image_file))
                all_labels.append(label)
                label_image_count += 1
        print(f"Loaded {label_image_count} images for label: {label}")
    if not all_image_paths:
        print("No images were loaded. Please check the provided folder paths.", file=sys.stderr)
        return pd.DataFrame(columns=['image_path', 'label'])
    
    data = pd.DataFrame({
        'image_path': all_image_paths,
        'label': all_labels
    })

    print("Data loading completed. Total images loaded:", len(data))
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data
    return data

### preprocess images (resize, normalize, etc.)
'''
def preprocess_images(data):
    print("Starting image preprocessing...")
'''


if __name__ == "__main__":
    ### data preperation / loading data
    base_path = "FruitClassification/data/fruits-360_100x100/fruits-360/Training/"
    paths = {
        "apple": [
            os.path.join(base_path, "Apple Red 1/"),
            os.path.join(base_path, "Apple Red 2/"),
            os.path.join(base_path, "Apple Red 3/")
        ],
        "banana": [
            os.path.join(base_path, "Banana 1/"),
            os.path.join(base_path, "Banana 3/"),
            os.path.join(base_path, "Banana 4/"),
            os.path.join(base_path, "Banana Lady Finger 1/")
        ]
    }
    data = load_data(paths)
    if not data.empty:
        print(data.head())
        print(data['label'].value_counts())
    else:
        print("No data loaded.")


    ### data processing / feature extraction (HOG, LBP, color histogram, etc. )