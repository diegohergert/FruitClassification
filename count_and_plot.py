import os
import pandas as pd
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
# POINT THIS TO THE PARENT FOLDER containing both 'Training' and 'Test'
BASE_DIR = r"c:\Users\DiegoPC\Documents\project\FruitClassification\data\fruits-360_100x100\fruits-360"
SUB_DIRS = ["Training", "Test"]
OUTPUT_CSV = "dataset_distribution.csv"

# ==========================================
# 2. DATA LISTS
# ==========================================

GROUPS = {
    "Healthy Apples": [
        "Apple 5", "Apple 10", "Apple 11", "Apple 14", "Apple 17",  
        "Apple 18", "Apple Braeburn 1", "Apple Crimson Snow 1", 
        "Apple Golden 2", "Apple Golden 3", "Apple Granny Smith 1", 
        "Apple Pink Lady 1", "Apple Red 1", "Apple Red 2", "Apple Red 3", 
        "Apple Red Delicious 1", "Apple Red Yellow 1"
    ],
    "Healthy Others": [
        "Peach 1", "Peach 2", "Peach Flat 1",
        "Pear 1", "Pear 2", "Pear 5", "Pear Forrelle 1", "Pear Monster 1", "Pear Red 1", "Pear Williams 1",
        "Tomato 1", "Tomato 2", "Tomato 3", "Tomato 4", "Tomato 5", "tomato 7", "Tomato 10", 
        "Tomato Cherry Maroon 1", "Tomato Cherry Orange 1", "Tomato Cherry Red 1",
        "Tomato Cherry Red 2", "Tomato Cherry Yellow 1", "Tomato Heart 1", "Tomato Yellow 1"
    ],
    "Healthy Bananas": [
        "Banana 1", "Banana 3", "Banana 4", "Banana Lady Finger 1"
    ],
    "Apple Anomalies": [
        "Apple Core 1", "Apple hit 1", "Apple Rotten 1"
    ],
    "Other Anomalies": [
        "Peach 3", "Peach 4", "Peach 5", "Peach 6",
        "Pear 3", "Pear 6", "Pear 7", "Pear 8", "Pear 12",
        "Tomato not Ripen 1"
    ]
}

# ==========================================
# 3. COUNTING LOGIC
# ==========================================

def get_combined_count(folder_name):
    """
    Looks in both 'Training' and 'Test' subdirectories.
    """
    total_count = 0
    valid_exts = {'.jpg', '.jpeg', '.png'}
    found = False
    
    for subdir in SUB_DIRS:
        folder_path = os.path.join(BASE_DIR, subdir, folder_name)
        
        if os.path.exists(folder_path):
            found = True
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if Path(file).suffix.lower() in valid_exts:
                        total_count += 1
                        
    return total_count if found else 0

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    print(f"Scanning Base Directory: {BASE_DIR}")
    
    data_rows = []

    # Loop through each group defined in the dictionary
    for category_name, folder_list in GROUPS.items():
        print(f"Processing {category_name}...")
        
        # Determine strict Type for easier graphing later
        is_anomaly = "Anomaly" in category_name
        fruit_type = "Healthy" if not is_anomaly else "Anomaly"

        for folder in folder_list:
            count = get_combined_count(folder)
            
            # Append data row
            data_rows.append({
                "Category": category_name,   # e.g., "Healthy Apples"
                "Subclass": folder,          # e.g., "Apple Red 1"
                "Count": count,              # e.g., 492
                "Type": fruit_type           # e.g., "Healthy" or "Anomaly"
            })

    # Convert to DataFrame
    df = pd.DataFrame(data_rows)
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*50)
    print(f"SUCCESS! Data saved to: {OUTPUT_CSV}")
    print("="*50)
    
    # Print a quick preview
    print("\nData Preview:")
    print(df.groupby("Category")["Count"].sum())

if __name__ == "__main__":
    main()