import pandas as pd
import os
from pathlib import Path
import sys
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from tqdm import tqdm
import joblib

## LOAD DATA
def load_data(base_paths):
    all_image_paths = []
    all_labels = []

    print("Starting data loading process...")
    
    if not isinstance(base_paths, list):
        base_paths = [base_paths]

    for base_dir in base_paths:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Warning: Base directory {base_path} does not exist. Skipping...", file=sys.stderr)
            continue
        
        print(f"Scanning for images in: {base_path}")
        image_files = list(base_path.glob('*/*.jpg'))
        
        if not image_files:
            print(f"Warning: No images found in subfolders of {base_path}", file=sys.stderr)
            continue

        for img_file in tqdm(image_files, desc=f"Finding files in {base_dir}"):
            all_image_paths.append(str(img_file))
            all_labels.append(img_file.parent.name) 
            
    if not all_image_paths:   
        print("No images were loaded. Please check the folder paths.", file=sys.stderr)
        return pd.DataFrame(columns=['image_path', 'label'])
    
    data = pd.DataFrame({
        'image_path': all_image_paths,
        'label': all_labels
    })

    print(f"Data loading completed. Total images loaded: {len(data)}")
    data = data.sample(frac=1).reset_index(drop=True) 
    return data


# EXTRACT FEATURES
IMG_SIZE = (100, 100)

###FEATURE RANGES FROM 0-256
def extract_hsv_histogram(img, bins = (8, 8, 8)):
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp_features(img, numPoints=24, radius=8, method="uniform"):
    lbp = local_binary_pattern(img, numPoints, radius, method=method)
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    hist = hist.astype("float")
    cv2.normalize(hist, hist)
    return hist

def extract_hog_features(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    hog_features = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, block_norm='L2-Hys', transform_sqrt=True, visualize=False)
    return hog_features

## preprocess
def preprocess_image(data, param_config, save_path):
    #read image
    hsv_features = []
    lbp_features = []
    hog_features = []

    hsv_params = param_config.get('hsv', {})
    lbp_params = param_config.get('lbp', {})
    hog_params = param_config.get('hog', {})

    for img_path in tqdm(data['image_path'], desc="Extracting features"):
        try:
            image = cv2.imread(img_path)
            if image is None:
                hsv_features.append(None)
                lbp_features.append(None)
                hog_features.append(None)
                continue
            image = cv2.resize(image, IMG_SIZE)
            hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            hsv_features.append(extract_hsv_histogram(hsv_img, **hsv_params))
            lbp_features.append(extract_lbp_features(gray_img, **lbp_params))
            hog_features.append(extract_hog_features(gray_img, **hog_params))

        except Exception as e:
            print(f"Error processing image {img_path}: {e}", file=sys.stderr)
            hsv_features.append(None)
            lbp_features.append(None)
            hog_features.append(None)
    
    processed_data = data.copy()
    processed_data['hsv_features'] = hsv_features
    processed_data['lbp_features'] = lbp_features
    processed_data['hog_features'] = hog_features

    # combine features
    def concat_feats(row, cols):
        try:
            return np.concatenate([row[col] for col in cols])
        except:
            return None

    processed_data['hog_lbp_features'] = processed_data.apply(
        lambda row: concat_feats(row, ['hog_features', 'lbp_features']), axis=1)
    processed_data['hog_hsv_features'] = processed_data.apply(
        lambda row: concat_feats(row, ['hog_features', 'hsv_features']), axis=1)
    processed_data['lbp_hsv_features'] = processed_data.apply(
        lambda row: concat_feats(row, ['lbp_features', 'hsv_features']), axis=1)
    processed_data['all_features'] = processed_data.apply(
        lambda row: concat_feats(row, ['hog_features', 'lbp_features', 'hsv_features']), axis=1)

    processed_data.dropna(inplace=True)
    processed_data.reset_index(drop=True, inplace=True)
    print("Feature extraction completed. Total images processed:", len(processed_data))
    data_to_save = {
        'parameters': param_config,
        'data': processed_data
    }
    joblib.dump(data_to_save, save_path)
    print(f"Processed data saved to {save_path}")
    return processed_data

if __name__ == "__main__":

    base_folder_paths = [
        "data/fruits-360_100x100/fruits-360/Training/",
        "data/fruits-360_100x100/fruits-360/Test/"
    ]

    data = load_data(base_folder_paths)
    if data.empty:
        print("No data to process. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    expirement_configs = [
        {
            "save_path": "params/feature_set_5.joblib",
            "param_config": {
                "hsv": {"bins": (4, 8, 2)},
                "lbp": {"numPoints": 24, "radius": 4, "method": "uniform"},
                "hog": {"orientations": 24, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}
            }
        },
        {
            "save_path": "params/feature_set_6.joblib",
            "param_config": {
                "hsv": {"bins": (4, 8, 2)},
                "lbp": {"numPoints": 24, "radius": 4, "method": "uniform"},
                "hog": {"orientations": 18, "pixels_per_cell": (8, 8), "cells_per_block": (3, 3)}
            }
        },
        {
            "save_path": "params/feature_set_7.joblib",
            "param_config": {
                "hsv": {"bins": (4, 8, 2)},
                "lbp": {"numPoints": 24, "radius": 2, "method": "uniform"},
                "hog": {"orientations": 18, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}
            }
        },
        {
            "save_path": "params/feature_set_8.joblib",
            "param_config": {
                "hsv": {"bins": (4, 8,2)},
                "lbp": {"numPoints": 24, "radius": 2, "method": "uniform"},
                "hog": {"orientations": 24, "pixels_per_cell": (8, 8), "cells_per_block": (3, 3)}
            }
        }
    ]


    for config in expirement_configs:
        print(f"Processing with config: {config['param_config']}")
        preprocess_image(data=data.copy(), param_config=config['param_config'], save_path=config['save_path'])

    print("All feature extraction experiments completed.")



