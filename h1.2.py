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
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
RESULTS_DIR = "results_h1.2/"
CACHE_DIR = "params_h1.2/"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DATA_DIRS = [
    "data/fruits-360_100x100/fruits-360/Training/",
    "data/fruits-360_100x100/fruits-360/Test/"
]

IMG_SIZE = (100, 100)

APPLE_SPECIES = [
    "Apple 5", "Apple 6", "Apple 7", "Apple 8", "Apple 9",
    "Apple 10", "Apple 11", "Apple 12", "Apple 13", "Apple 14",
    "Apple 15", "Apple 16", "Apple 17", "Apple 18", "Apple 19",
    "Apple Braeburn 1", "Apple Crimson Snow 1",
    "Apple Golden 1", "Apple Golden 2", "Apple Golden 3",
    "Apple Granny Smith 1", "Apple Pink Lady 1",
    "Apple Red 1", "Apple Red 2", "Apple Red 3",
    "Apple Red Delicious 1", "Apple Red Yellow 1", "Apple Red Yellow 2"
]

FEAT_COLUMNS = [
    "hsv_features", "lbp_features", "hog_features",
    "hog_lbp_features", "hog_hsv_features", "lbp_hsv_features", 
    "all_features"
]

def get_configs():
    return [
        {"name": "feature_set_1", "hsv_bins": (8, 8, 8), "lbp_p": 8, "lbp_r": 1, "hog_orient": 9, "hog_cell": (8, 8)},
        {"name": "feature_set_2", "hsv_bins": (16, 16, 16), "lbp_p": 32, "lbp_r": 4, "hog_orient": 12, "hog_cell": (4, 4)},
        {"name": "feature_set_3", "hsv_bins": (8, 12, 3), "lbp_p": 16, "lbp_r": 4, "hog_orient": 9, "hog_cell": (16, 16)},
        {"name": "feature_set_4", "hsv_bins": (16, 16, 16), "lbp_p": 24, "lbp_r": 3, "hog_orient": 9, "hog_cell": (8, 8)},
        {"name": "feature_set_5", "hsv_bins": (4, 8, 2), "lbp_p": 24, "lbp_r": 4, "hog_orient": 24, "hog_cell": (8, 8)},
        {"name": "feature_set_6", "hsv_bins": (4, 8, 2), "lbp_p": 24, "lbp_r": 4, "hog_orient": 18, "hog_cell": (8, 8)},
        {"name": "feature_set_7", "hsv_bins": (4, 8, 2), "lbp_p": 24, "lbp_r": 2, "hog_orient": 18, "hog_cell": (8, 8)},
        {"name": "feature_set_8", "hsv_bins": (4, 8, 2), "lbp_p": 24, "lbp_r": 2, "hog_orient": 24, "hog_cell": (8, 8)}
    ]
CONFIGS = get_configs()

# --- FEATURE EXTRACTION ---
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
    path_obj = Path(f)
    label = path_obj.parent.name
    
    # Filter for H2 Apples only to save time/space
    if label not in APPLE_SPECIES: return None 

    img = cv2.imread(str(f))
    if img is None: return None
    img = cv2.resize(img, IMG_SIZE)
    
    hsv_stat = extract_hsv_stats(img)
    h_hist = extract_hsv_hist(img, cfg['hsv_bins'])
    hsv_all = np.concatenate([hsv_stat, h_hist])
    
    lbp = extract_lbp(img, cfg['lbp_p'], cfg['lbp_r'])
    hg = extract_hog(img, cfg['hog_orient'], cfg['hog_cell'])
    
    return {
        'image_path': str(f),
        'label': label,
        'hsv_features': hsv_all,
        'lbp_features': lbp,
        'hog_features': hg,
        'hog_lbp_features': np.concatenate([hg, lbp]),
        'hog_hsv_features': np.concatenate([hg, hsv_all]),
        'lbp_hsv_features': np.concatenate([lbp, hsv_all]),
        'all_features': np.concatenate([hg, lbp, hsv_all])
    }

def load_specific_config(cfg):
    filename = f"{cfg['name']}.joblib"
    filepath = os.path.join(CACHE_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"Loading cached: {filename}")
        return joblib.load(filepath)
    
    print(f"Extracting features for {cfg['name']}...")
    all_files = []
    for d in DATA_DIRS:
        path = Path(d)
        if path.exists():
            all_files.extend(list(path.glob('*/*.jpg')))
            
    results = Parallel(n_jobs=-1)(
        delayed(process_image)(f, cfg) for f in tqdm(all_files, desc="Extraction")
    )
    
    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows)
    
    data_to_save = {'parameters': cfg, 'data': df}
    print(f"Saving {filename}...")
    joblib.dump(data_to_save, filepath)
    return data_to_save

# --- PIPELINE ---
def plot_umap(X, y, title, filename):
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=y, s=15, palette="tab20", legend=False)
    plt.title(title)
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def run_pipeline():
    kmeans_log = []
    hdbscan_log = []

    for cfg in CONFIGS:
        print(f"\n=== Processing {cfg['name']} ===")
        loaded = load_specific_config(cfg)
        df_full = loaded['data']
        
        # Ensure we only have apples (redundant check if process_image filtered, but safe)
        df = df_full[df_full['label'].isin(APPLE_SPECIES)].copy()
        if df.empty: continue

        for feat_col in FEAT_COLUMNS:
            X_raw = np.stack(df[feat_col].values)
            y_true = df['label'].values
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_raw)
            
            pca = PCA(n_components=0.95, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            for trans_name, X_data in [("Original", X_scaled), ("PCA", X_pca)]:
                
                # 1. K-Means
                for k in [2, 10, 20, 29, 35]:
                    km = KMeans(n_clusters=k, random_state=42, n_init=3)
                    y_pred = km.fit_predict(X_data)
                    kmeans_log.append({
                        "Config": cfg['name'], "Feature": feat_col, "Transform": trans_name,
                        "K": k, "Silhouette": silhouette_score(X_data, y_pred),
                        "Inertia": km.inertia_
                    })

                # 2. HDBSCAN
                hdb = HDBSCAN(min_cluster_size=12)
                y_pred_hdb = hdb.fit_predict(X_data)
                
                non_noise = y_pred_hdb != -1
                if np.sum(non_noise) > 10:
                    ari = adjusted_rand_score(y_true[non_noise], y_pred_hdb[non_noise])
                    sil = silhouette_score(X_data[non_noise], y_pred_hdb[non_noise])
                else:
                    ari = 0; sil = 0

                hdbscan_log.append({
                    "Config": cfg['name'], "Feature": feat_col, "Transform": trans_name,
                    "ARI": ari, "Silhouette": sil, "Clusters": len(set(y_pred_hdb))-1
                })
                
                if ari > 0.8:
                    plot_umap(X_data, y_true, f"ARI={ari:.3f} {cfg['name']}", f"UMAP_{cfg['name']}_{feat_col}_{trans_name}.png")

        gc.collect()

    pd.DataFrame(kmeans_log).to_csv(os.path.join(RESULTS_DIR, "h2_kmeans_results.csv"), index=False)
    df_hdb = pd.DataFrame(hdbscan_log).sort_values(by="ARI", ascending=False)
    df_hdb.to_csv(os.path.join(RESULTS_DIR, "h2_hdbscan_results.csv"), index=False)
    
    print("\n=== TOP HDBSCAN RESULTS ===")
    print(df_hdb.head())

if __name__ == "__main__":
    run_pipeline()