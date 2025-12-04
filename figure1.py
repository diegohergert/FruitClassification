import matplotlib.pyplot as plt
import cv2
import os
import random
from pathlib import Path
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "data/fruits-360_100x100/fruits-360/Training/"

ROWS_CONFIG = [
    {
        "title": "Hypothesis 1.1: Macro-Separation",
        "left": "Apple Red 1",  "left_name": "Class A: Apple",
        "right": "Banana 3",    "right_name": "Class B: Banana"
    },
    {
        "title": "Hypothesis 1.2: Micro-Separation",
        "left": "Apple Red 1",          "left_name": "Species A: Red 1",
        "right": "Apple Red Delicious 1", "right_name": "Species B: Red Del."
    },
    {
        "title": "Hypothesis 2.1: Apple Anomaly Detection",
        "left": "Apple Red 1",    "left_name": "Normal Apple",
        "right": "Apple Rotten 1", "right_name": "Defect: Rot"
    },
    {
        "title": "Hypothesis 2.2: Mixed Fruit Anomaly Detection",
        "left": "Pear 1",     "left_name": "Normal Pear",
        "right": "Pear 7",    "right_name": "Defect: Pear Anomaly" 
    }
]

def get_n_random_images(folder_name, n=2):
    path = os.path.join(DATA_DIR, folder_name)
    placeholder = np.zeros((100, 100, 3), dtype=np.uint8) + 255 # White background
    
    if not os.path.exists(path):
        return [placeholder] * n
    
    files = list(Path(path).glob("*.jpg"))
    if not files:
        return [placeholder] * n
        
    if len(files) >= n:
        selected_files = random.sample(files, n)
    else:
        selected_files = [random.choice(files) for _ in range(n)]
        
    images = []
    for p in selected_files:
        img = cv2.imread(str(p))
        if img is None: 
            images.append(placeholder)
        else:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images

def create_figure():
    # Create Layout
    fig, axes = plt.subplots(4, 4, figsize=(10, 14))
    
    # Adjust spacing
    # hspace=0.6 provides vertical room for titles
    # wspace=0.05 keeps images in a pair close together
    plt.subplots_adjust(hspace=0.6, wspace=0.05, top=0.90, bottom=0.05)
    
    for row_idx, config in enumerate(ROWS_CONFIG):
        imgs_left = get_n_random_images(config["left"], 2)
        imgs_right = get_n_random_images(config["right"], 2)
        
        # --- Plot Images ---
        for i, img in enumerate(imgs_left):
            axes[row_idx, i].imshow(img)
            axes[row_idx, i].axis("off")
            
        for i, img in enumerate(imgs_right):
            axes[row_idx, i+2].imshow(img)
            axes[row_idx, i+2].axis("off")

        # --- Calculate Label Positions Dynamicallly ---
        # We need to draw text relative to the figure, not the axes, to span columns.
        
        # Get Bounding Boxes of the axes to find centers
        bbox_0 = axes[row_idx, 0].get_position() # Col 0
        bbox_1 = axes[row_idx, 1].get_position() # Col 1
        bbox_2 = axes[row_idx, 2].get_position() # Col 2
        bbox_3 = axes[row_idx, 3].get_position() # Col 3
        
        # Center of Left Pair = Average of (Col 0 Left Edge) and (Col 1 Right Edge)
        left_center_x = (bbox_0.x0 + bbox_1.x1) / 2
        
        # Center of Right Pair
        right_center_x = (bbox_2.x0 + bbox_3.x1) / 2
        
        # Y Positions
        label_y = bbox_0.y1 + 0.015       # Slightly above image
        title_y = bbox_0.y1 + 0.055       # Above the labels
        
        # --- Add Text ---
        # 1. Class Labels (Centered over pairs)
        fig.text(left_center_x, label_y, config["left_name"], 
                 ha='center', va='bottom', fontsize=10)
        
        fig.text(right_center_x, label_y, config["right_name"], 
                 ha='center', va='bottom', fontsize=10)
        
        # 2. Main Row Title (Centered over whole row)
        fig.text(0.5, title_y, config["title"], 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    save_path = "Figure_1_Centered_Labels.png"
    plt.savefig(save_path, dpi=300) # Removed bbox_inches='tight' to respect manual spacing
    print(f"Figure saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    create_figure()