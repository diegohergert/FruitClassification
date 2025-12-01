import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import sys
import gc 
from pathlib import Path
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog
from joblib import Parallel, delayed

# --- METRIC IMPORTS ---
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score

# =============================================================================
#   1. CONFIGURATION
# =============================================================================

OUTPUT_DIR = "results_h1.1/"
CACHE_DIR = "params_h1.1/" # Separate cache for H1 to avoid overwriting H3/H2
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DATA_DIRS = [
    "data/fruits-360_100x100/fruits-360/Training/",
    "data/fruits-360_100x100/fruits-360/Test/"
]

# H1 Specific Lists
APPLES = [
    "Apple 5", "Apple 6", "Apple 7", "Apple 8", "Apple 9",
    "Apple 10", "Apple 11", "Apple 12", "Apple 13", "Apple 14",
    "Apple 15", "Apple 16", "Apple 17", "Apple 18", "Apple 19",
    "Apple Braeburn 1", "Apple Crimson Snow 1",
    "Apple Golden 1", "Apple Golden 2", "Apple Golden 3",
    "Apple Granny Smith 1", "Apple Pink Lady 1",
    "Apple Red 1", "Apple Red 2", "Apple Red 3",
    "Apple Red Delicious 1", "Apple Red Yellow 1", "Apple Red Yellow 2"
]

BANANAS = [
    "Banana", "Banana Lady Finger", "Banana Red"
]

IMG_SIZE = (100, 100)
SCALERS_TO_TEST = ["MinMax"] 

# =============================================================================
#   2. CONFIG DEFINITIONS
# =============================================================================

def get_configs():
    configs = []
    # Using your best config based on previous chats
    configs.append({
        "name": "Config_v23.1",
        "hsv_bins": (24, 24, 24),
        "lbp_p": 24, "lbp_r": 2, 
        "hog_orient": 12, "hog_cell": (12, 12)
    })
    return configs
CONFIGS = get_configs()

# =============================================================================
#   3. FEATURE EXTRACTION (The .joblib creation)
# =============================================================================

def extract_hsv_stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean, std = cv2.meanStdDev(hsv)
    return np.concatenate([mean.flatten(), std.flatten()])

def extract_hsv_hist(img, bins):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist) 
    return hist.flatten()

def extract_lbp(img, P, R):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float") 
    cv2.normalize(hist, hist)
    return hist

def extract_hog(img, orient, cell):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(gray, orientations=orient, pixels_per_cell=cell, 
               cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True, visualize=False)

def process_image(f, cfg):
    """
    Worker function. Filters for Apples AND Bananas.
    """
    path_obj = Path(f)
    label = path_obj.parent.name
    
    # H1 Filter: Must be in Apple list or Banana list
    is_apple = label in APPLES
    is_banana = label in BANANAS

    if not is_apple and not is_banana:
        return None
    
    img = cv2.imread(str(f))
    if img is None: return None
    img = cv2.resize(img, IMG_SIZE)
    
    # Extract
    stats = extract_hsv_stats(img)
    h_hist = extract_hsv_hist(img, cfg['hsv_bins'])
    lbp = extract_lbp(img, cfg['lbp_p'], cfg['lbp_r'])
    hg = extract_hog(img, cfg['hog_orient'], cfg['hog_cell'])
    
    vector = np.concatenate([stats, h_hist, lbp, hg])
    
    # Return Broad Label for H1 (Apple vs Banana)
    broad_label = "Apple" if is_apple else "Banana"

    return {
        'image_path': str(f),
        'label': label,
        'broad_label': broad_label, # Helper for H1
        'features': vector
    }

def load_specific_config(cfg):
    filename = f"feat_H1_{cfg['name']}.joblib"
    filepath = os.path.join(CACHE_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"Loading cached features: {filename}")
        return joblib.load(filepath)
    
    print(f"Extracting features for {cfg['name']} (Apples + Bananas)...")
    
    all_files = []
    for d in DATA_DIRS:
        path = Path(d)
        if path.exists():
            all_files.extend(list(path.glob('*/*.jpg')))
            
    results = Parallel(n_jobs=-1)(
        delayed(process_image)(f, cfg) for f in tqdm(all_files, desc="Parallel Extraction")
    )
    
    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows)
    
    print(f"Saving {filename}...")
    joblib.dump(df, filepath)
    return df

# =============================================================================
#   4. BALANCING LOGIC (Phase 2.1)
# =============================================================================

def balance_dataset(df):
    """
    Ensures we have equal numbers of Apples and Bananas to prevent
    cluster bias towards the majority class.
    """
    g = df.groupby('broad_label')
    min_size = g.size().min()
    
    print(f"   > Balancing data to {min_size} samples per class...")
    balanced_df = g.apply(lambda x: x.sample(min_size, random_state=42)).reset_index(drop=True)
    return balanced_df

# =============================================================================
#   5. PIPELINE
# =============================================================================

def run_pipeline():
    results = []
    
    for cfg in CONFIGS:
        print(f"\n=== Processing Config: {cfg['name']} ===")
        
        # 1. Load Data (Joblib creation happens here if needed)
        curr_df = load_specific_config(cfg)
        
        # 2. Balance Data (H1 Requirement)
        curr_df = balance_dataset(curr_df)
        
        X_raw = np.stack(curr_df['features'].values)
        y_true = curr_df['broad_label'].values
        
        # Map labels to int for ARI
        y_encoded = pd.Series(y_true).map({'Apple': 0, 'Banana': 1}).values

        for scaler_name in SCALERS_TO_TEST:
            exp_id = f"{cfg['name']}_{scaler_name}"
            print(f"   > Running Experiment: {exp_id}")
            
            # 3. Preprocessing
            if scaler_name == "MinMax": scaler = MinMaxScaler()
            else: scaler = StandardScaler()
            
            X_scaled = scaler.fit_transform(X_raw)
            
            # 4. PCA (Phase 2.1)
            pca = PCA(n_components=0.95, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            # 5. Model: K-Means (Phase 3)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            y_pred = kmeans.fit_predict(X_pca)
            
            # 6. Metrics
            ari = adjusted_rand_score(y_encoded, y_pred)
            sil = silhouette_score(X_pca, y_pred)
            ch_score = calinski_harabasz_score(X_pca, y_pred)
            inertia = kmeans.inertia_
            
            # Log
            res = {
                "Config": cfg['name'],
                "Scaler": scaler_name,
                "PCA_Vars": X_pca.shape[1],
                "ARI": ari,
                "Silhouette": sil,
                "Calinski": ch_score,
                "Inertia": inertia
            }
            results.append(res)
            print(f"     > ARI: {ari:.4f} | Silhouette: {sil:.4f}")

        # Memory Cleanup
        del curr_df, X_raw, X_scaled, X_pca
        gc.collect()

    # Save Final Results
    res_df = pd.DataFrame(results).sort_values(by="ARI", ascending=False)
    res_df.to_csv(os.path.join(OUTPUT_DIR, "h1_final_results.csv"), index=False)
    print("\n=== FINAL H1 RANKING ===")
    print(res_df)

if __name__ == "__main__":
    run_pipeline()