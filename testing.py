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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, 
    precision_recall_curve, average_precision_score,
    f1_score, recall_score
)

# =============================================================================
#   1. CONFIGURATION
# =============================================================================

OUTPUT_DIR = "h3_memory_optimized_results/v6/"
CACHE_DIR = "paramsv2/" # We will save separate files here
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DATA_DIRS = [
    "data/fruits-360_100x100/fruits-360/Training/",
    "data/fruits-360_100x100/fruits-360/Test/"
]

APPLES = [
    "Apple 5", "Apple 10", "Apple 11", "Apple 14", "Apple 17",  
    "Apple 18", "Apple Braeburn 1", "Apple Crimson Snow 1", 
    "Apple Golden 2", "Apple Golden 3", "Apple Granny Smith 1", 
    "Apple Pink Lady 1", "Apple Red 1", "Apple Red 2", "Apple Red 3", 
    "Apple Red Delicious 1", "Apple Red Yellow 1"
]
ANOMALIES = ["Apple Core 1", "Apple hit 1", "Apple Rotten 1"]

IMG_SIZE = (100, 100)
SCALERS_TO_TEST = ["MinMax"]

# =============================================================================
#   2. CONFIG DEFINITIONS
# =============================================================================

def get_configs():
    configs = []
    configs.append({
        "name": "Config_v8",
        "hsv_bins": (16, 16, 16),
        "lbp_p": 24, "lbp_r": 2, 
        "hog_orient": 12, "hog_cell": (12, 12)
    })

    configs.append({
        "name": "Config_v11",
        "hsv_bins": (16, 16, 16),
        "lbp_p": 28, "lbp_r": 2, 
        "hog_orient": 12, "hog_cell": (12, 12)
    })

    configs.append({
        "name": "Config_v10",
        "hsv_bins": (12, 12, 12),
        "lbp_p": 24, "lbp_r": 2, 
        "hog_orient": 12, "hog_cell": (12, 12)
    })

    configs.append({
        "name": "Config_v12",
        "hsv_bins": (24, 24, 24),
        "lbp_p": 24, "lbp_r": 2, 
        "hog_orient": 12, "hog_cell": (12, 12)
    })
    return configs
CONFIGS = get_configs()

# =============================================================================
#   3. FEATURE EXTRACTION
# =============================================================================

def extract_hsv_stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean, std = cv2.meanStdDev(hsv)
    return np.concatenate([mean.flatten(), std.flatten()])

def extract_hsv_hist(img, bins):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist) #test this non inplace
    return hist.flatten()

def extract_lbp(img, P, R):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float") #test this non inplace
    cv2.normalize(hist, hist)
    return hist

def extract_hog(img, orient, cell):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(gray, orientations=orient, pixels_per_cell=cell, 
               cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True, visualize=False)

def process_image(f, cfg):
    """
    Helper function to process a single image. 
    Returns a dictionary row or None if the image is invalid.
    """
    # Global constants (IMG_SIZE, etc.) are accessible here
    path_obj = Path(f)
    label = path_obj.parent.name
    
    # Filter labels
    if label not in APPLES and label not in ANOMALIES: 
        return None
    
    img = cv2.imread(str(f))
    if img is None: 
        return None
    
    img = cv2.resize(img, IMG_SIZE)
    
    # Extract features
    stats = extract_hsv_stats(img)
    h_hist = extract_hsv_hist(img, cfg['hsv_bins'])
    lbp = extract_lbp(img, cfg['lbp_p'], cfg['lbp_r'])
    hg = extract_hog(img, cfg['hog_orient'], cfg['hog_cell'])
    
    vector = np.concatenate([stats, h_hist, lbp, hg])
    
    return {
        'image_path': str(f),
        'label': label,
        'features': vector
    }

def load_specific_config(cfg):
    filename = f"feat_{cfg['name']}.joblib"
    filepath = os.path.join(CACHE_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"Loading cached features: {filename}")
        return joblib.load(filepath)
    
    print(f"Extracting features for {cfg['name']}...")
    
    # 1. Gather all file paths first
    all_files = []
    for d in DATA_DIRS:
        path = Path(d)
        if path.exists():
            all_files.extend(list(path.glob('*/*.jpg')))
            
    # 2. Run in Parallel
    # n_jobs=-1 uses all available CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(process_image)(f, cfg) for f in tqdm(all_files, desc="Parallel Extraction")
    )
    
    # 3. Filter out None values (from invalid images or ignored labels)
    rows = [r for r in results if r is not None]
            
    df = pd.DataFrame(rows)
    print(f"Saving {filename}...")
    joblib.dump(df, filepath)
    return df

# =============================================================================
#   4. VISUALIZATION
# =============================================================================

def save_image_grid(image_paths, title, filename, n=16):
    images = []
    for p in image_paths[:n]:
        img = cv2.imread(p)
        if img is not None: images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if not images: return
    
    cols = 4
    rows = int(np.ceil(len(images) / cols))
    
    # Create the grid
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
    fig.suptitle(title, fontsize=14)
    
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')
        else: 
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_confusion_matrix_custom(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Healthy', 'Anomaly'])
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_sensitivity_analysis(model, X_train_p, X_test_p, y_test, title, filename):
    """
    Plots how Test F1 Score changes based on 'Strictness' (Training Percentile).
    This proves if the model is robust or just lucky.
    """
    train_scores = -1 * model.decision_function(X_train_p)
    test_scores = -1 * model.decision_function(X_test_p)
    
    percentiles = np.linspace(0.5, 10, 20) # Check top 0.5% to top 10% strictness
    f1_scores = []
    
    for p in percentiles:
        # Define threshold based strictly on TRAINING data (Safe!)
        thresh = np.percentile(train_scores, 100 - p) # (High score = anomaly)
        
        y_pred = (test_scores > thresh).astype(int)
        f1_scores.append(f1_score(y_test, y_pred))
        
    plt.figure(figsize=(8, 5))
    plt.plot(percentiles, f1_scores, marker='o')
    plt.xlabel("Assumed Contamination % (Training Threshold)")
    plt.ylabel("Test F1 Score")
    plt.title(f"Sensitivity Analysis: {title}")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_score_distribution(y_test, test_scores, title, filename):
    plt.figure(figsize=(10, 6))
    
    # Healthy scores
    sns.kdeplot(test_scores[y_test == 0], fill=True, color='green', label='Healthy (Test)')
    
    # Anomaly scores
    sns.kdeplot(test_scores[y_test == 1], fill=True, color='red', label='Anomaly (Test)')
    
    plt.title(f"Score Separation: {title}")
    plt.xlabel("Anomaly Score (Higher is worse)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def precision_at_k(y_true, scores, k=100):
    # Sort by score (descending)
    ranked_indices = np.argsort(scores)[::-1]
    top_k_indices = ranked_indices[:k]
    # How many of the top K were actually 1 (anomaly)?
    return np.sum(y_true[top_k_indices]) / k
# =============================================================================
#   5. PIPELINE
# =============================================================================

def run_pipeline():
    results = []
    
    # LOOP CONFIGS
    for cfg in CONFIGS:
        print(f"\n=== Processing Config: {cfg['name']} ===")
        
        # 1. LOAD DATA
        curr_df = load_specific_config(cfg)
        
        # 2. Loop Scalers
        for scaler_name in SCALERS_TO_TEST:
            exp_id = f"{cfg['name']}_{scaler_name}"
            print(f"   > Running Experiment: {exp_id}")
            
            # Initialize Log (Added P@100)
            metrics_log = {"AP": [], "F1": [], "Recall": [], "ROC": [], "P@100": []}
            
            # 3. Stability Loop
            for seed in [67, 69, 123, 420, 42]:
                # Split Data
                df_healthy = curr_df[curr_df['label'].isin(APPLES)]
                df_anom = curr_df[curr_df['label'].isin(ANOMALIES)]
                
                train_healthy, test_healthy = train_test_split(df_healthy, test_size=0.3, random_state=seed)
                
                X_train = np.stack(train_healthy['features'].values)
                X_test_healthy = np.stack(test_healthy['features'].values)
                X_test_anom = np.stack(df_anom['features'].values)
                X_test = np.concatenate([X_test_healthy, X_test_anom])
                
                y_test = np.concatenate([np.zeros(len(X_test_healthy)), np.ones(len(X_test_anom))])
                test_paths = np.concatenate([test_healthy['image_path'].values, df_anom['image_path'].values])
                
                # Preprocessing
                if scaler_name == "MinMax": scaler = MinMaxScaler()
                else: scaler = StandardScaler()
                
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                
                pca = PCA(n_components=0.95, random_state=seed)
                X_train_p = pca.fit_transform(X_train_s)
                X_test_p = pca.transform(X_test_s)
                
                # Model
                model = IsolationForest(contamination='auto', random_state=seed, n_jobs=-1)
                model.fit(X_train_p)
                
                # SCORES: Inverted so Higher = Anomaly
                test_scores = -1 * model.decision_function(X_test_p)
                
                # --- METRICS CALCULATION ---
                
                # 1. Safe "Curve" Metrics (Independent of threshold)
                ap = average_precision_score(y_test, test_scores)
                roc = roc_auc_score(y_test, test_scores)
                p_at_100 = precision_at_k(y_test, test_scores, k=100)
                
                # 2. Safe "Hard" Metrics (F1 / Recall)
                # Determine Threshold from TRAINING data (No Peeking!)
                # We assume the top 1% of training scores might be noise/outliers
                train_scores = -1 * model.decision_function(X_train_p)
                threshold = np.percentile(train_scores, 99) 
                
                y_pred = (test_scores > threshold).astype(int)
                
                # Log them
                metrics_log['AP'].append(ap)
                metrics_log['ROC'].append(roc)
                metrics_log['P@100'].append(p_at_100)
                metrics_log['F1'].append(f1_score(y_test, y_pred))
                metrics_log['Recall'].append(recall_score(y_test, y_pred))
                
                # --- VISUALS (Seed 42 Only) ---
                if seed == 42:
                    plot_confusion_matrix_custom(y_test, y_pred, f"{exp_id} CM", f"{exp_id}_CM.png")
                    
                    # Sensitivity Plot (Does F1 crash if we change threshold?)
                    plot_sensitivity_analysis(model, X_train_p, X_test_p, y_test, exp_id, f"{exp_id}_sensitivity.png")
                    
                    # Distribution Plot (Are the mountains separated?)
                    plot_score_distribution(y_test, test_scores, exp_id, f"{exp_id}_dist.png")
                    
                    # Save FP/FN images
                    fp_paths = test_paths[(y_test == 0) & (y_pred == 1)]
                    save_image_grid(fp_paths, f"{exp_id} False Positives", f"{exp_id}_FP.png")
                    
                    fn_paths = test_paths[(y_test == 1) & (y_pred == 0)]
                    save_image_grid(fn_paths, f"{exp_id} False Negatives", f"{exp_id}_FN.png")

            # Log Results
            res = {
                "Config": cfg['name'],
                "Scaler": scaler_name,
                "AP_Mean": np.mean(metrics_log['AP']),
                "ROC_Mean": np.mean(metrics_log['ROC']),
                "F1_Mean": np.mean(metrics_log['F1']),
                "P@100_Mean": np.mean(metrics_log['P@100']) # Fixed this key
            }
            results.append(res)
            print(f"   > Avg AP: {res['AP_Mean']:.4f} | P@100: {res['P@100_Mean']:.4f}")
        
        # --- MEMORY CLEANUP ---
        del curr_df
        gc.collect() 
        print(f"   [Memory Cleared for {cfg['name']}]")

    # Save Final CSV
    res_df = pd.DataFrame(results).sort_values(by="AP_Mean", ascending=False)
    res_df.to_csv(os.path.join(OUTPUT_DIR, "final_results_ram_optimized.csv"), index=False)
    print("\n=== FINAL RANKING ===")
    print(res_df)

if __name__ == "__main__":
    run_pipeline()