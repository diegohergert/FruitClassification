import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Stop thread error 
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import sys
import cv2
import os
from sklearn.ensemble import IsolationForest
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- CONFIGURATION ---
Label = { 
    "apple_species": [
        "Apple 5", "Apple 8", "Apple 9/",
        "Apple 10", "Apple 11", "Apple 14",
        "Apple 15", "Apple 16", "Apple 17", "Apple 18",
        "Apple Braeburn 1", "Apple Crimson Snow 1",
        "Apple Pink Lady 1",
        "Apple Red 1", "Apple Red 2", "Apple Red 3",
        "Apple Red Delicious 1", "Apple Red Yellow 1", "Apple Red Yellow 2" 
    ],
    "apple_anomalies": [
        "Apple Core 1", "Apple hit 1", "Apple Rotten 1"
    ]
}

RESULTS_DIR = "results_h3_pca_test/"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_data_sample(data, feature_column, contamination=0.05, max_samples=20000):
    """Returns the RAW (Unscaled) X data and y_true labels."""
    healthy_df = data[data['label'].isin(Label["apple_species"])].copy()
    anomaly_df = data[data['label'].isin(Label["apple_anomalies"])].copy()
    
    n_healthy = min(len(healthy_df), int(max_samples * (1 - contamination)))
    n_anomalies = int(n_healthy * (contamination / (1 - contamination)))
    
    if n_anomalies > len(anomaly_df):
        n_anomalies = len(anomaly_df)
    
    print(f"  Dataset: {n_healthy} Healthy, {n_anomalies} Anomalies")
    
    healthy_sample = healthy_df.sample(n=n_healthy, random_state=42)
    anomaly_sample = anomaly_df.sample(n=n_anomalies, random_state=42)
    
    combined_df = pd.concat([healthy_sample, anomaly_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Return raw numpy array
    X_raw = np.array(combined_df[feature_column].tolist())
    y_true = combined_df['label'].apply(lambda x: 1 if x in Label["apple_species"] else -1).values
    
    return X_raw, y_true, combined_df

def visualize_predictions(y_true, y_pred, df, title_prefix, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy (1)", "Anomaly (-1)"])
    plt.figure(figsize=(6, 5))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{title_prefix} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path + "_cm.png")
    plt.close()

def plot_umap_anomalies(X_data, y_true, y_pred, save_path):
    if len(X_data) > 5000:
        idx = np.random.choice(len(X_data), 5000, replace=False)
        X_viz = X_data[idx]
        y_t_viz = y_true[idx]
        y_p_viz = y_pred[idx]
    else:
        X_viz = X_data
        y_t_viz = y_true
        y_p_viz = y_pred

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_viz)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=y_t_viz, palette={1: 'blue', -1: 'red'}, ax=axes[0], s=15, alpha=0.6)
    axes[0].set_title("Ground Truth (Red=Anomaly)")
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=y_p_viz, palette={1: 'blue', -1: 'red'}, ax=axes[1], s=15, alpha=0.6)
    axes[1].set_title("Prediction (Red=Anomaly)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    FEATURE_SET_PATH = [
        "params/feature_set_2.joblib", 
        "params/feature_set_3.joblib",
        "params/feature_set_1.joblib",
        "params/feature_set_4.joblib",
        "params/feature_set_7.joblib"
    ]
    
    FEAT_COLUMNS = ["lbp_hsv_features", "hog_hsv_features", "all_features"]
    CONTAMINATION = 0.02 
    
    SCALERS = {
        "MinMax": MinMaxScaler(),
        "Standard": StandardScaler()
    }
    
    # --- UPDATED: COMPARE VARIANCES ---
    PCA_VARIANCES = [0.9, 0.95, 0.99] 

    h3_results = []

    print("Starting PCA Variance Testing (95% vs 99%)...")

    for file_path in FEATURE_SET_PATH:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path}...")
            continue
            
        print(f"\nLoading {file_path}...")
        loaded = joblib.load(file_path)
        data = loaded['data']
        file_id = os.path.basename(file_path).replace(".joblib", "")

        for feat_col in FEAT_COLUMNS:
            print(f"  Feature: {feat_col}")
            
            # 1. Get Raw Data
            X_raw, y_true, df_subset = get_data_sample(data, feat_col, contamination=CONTAMINATION)
            
            # 2. Loop through Scalers
            for scaler_name, scaler in SCALERS.items():
                X_scaled = scaler.fit_transform(X_raw)
                
                # 3. Loop through PCA Variances
                for variance in PCA_VARIANCES:
                    print(f"      > Scaler: {scaler_name} | PCA Variance: {variance}")
                    
                    # Apply PCA
                    n_feats = X_scaled.shape[1]
                    if n_feats < 2: 
                        X_final = X_scaled 
                        pca_status = f"PCA_Skipped"
                    else:
                        pca = PCA(n_components=variance, random_state=42)
                        X_final = pca.fit_transform(X_scaled)
                        pca_status = f"PCA_{variance}"
                        print(f"        -> Reduced dims: {n_feats} -> {X_final.shape[1]}")

                    # --- MODEL 1: ISOLATION FOREST ---
                    iso = IsolationForest(contamination=CONTAMINATION, random_state=42, n_jobs=-1)
                    y_pred_iso = iso.fit_predict(X_final)
                    
                    report_iso = classification_report(y_true, y_pred_iso, output_dict=True, zero_division=0)
                    
                    # ID and Save
                    exp_id = f"{file_id}_{feat_col}_{scaler_name}_{pca_status}"
                    base_name = os.path.join(RESULTS_DIR, f"{exp_id}_ISO")
                    
                    # Visuals
                    plot_umap_anomalies(X_final, y_true, y_pred_iso, f"{base_name}_UMAP.png")
                    visualize_predictions(y_true, y_pred_iso, df_subset, f"ISO {pca_status}", base_name)
                    
                    h3_results.append({
                        "ParamSet": file_id,
                        "Feature": feat_col,
                        "Scaler": scaler_name,
                        "PCA_Var": variance,
                        "Model": "Isolation Forest",
                        "Dims_Retained": X_final.shape[1],
                        "F1_Anomaly": report_iso.get('-1', {}).get('f1-score', 0),
                        "Recall_Anomaly": report_iso.get('-1', {}).get('recall', 0),
                        "Precision_Anomaly": report_iso.get('-1', {}).get('precision', 0),
                    })

                    # --- MODEL 2: HDBSCAN ---
                    try:
                        hdb = HDBSCAN(min_cluster_size=2, min_samples=5)
                        y_pred_hdb_raw = hdb.fit_predict(X_final)
                        y_pred_hdb = np.where(y_pred_hdb_raw == -1, -1, 1)
                        
                        report_hdb = classification_report(y_true, y_pred_hdb, output_dict=True, zero_division=0)
                        
                        base_name_hdb = os.path.join(RESULTS_DIR, f"{exp_id}_HDB")
                        plot_umap_anomalies(X_final, y_true, y_pred_hdb, f"{base_name_hdb}_UMAP.png")
                        visualize_predictions(y_true, y_pred_hdb, df_subset, f"HDB {pca_status}", base_name_hdb)

                        h3_results.append({
                            "ParamSet": file_id,
                            "Feature": feat_col,
                            "Scaler": scaler_name,
                            "PCA_Var": variance,
                            "Model": "HDBSCAN",
                            "Dims_Retained": X_final.shape[1],
                            "F1_Anomaly": report_hdb.get('-1', {}).get('f1-score', 0),
                            "Recall_Anomaly": report_hdb.get('-1', {}).get('recall', 0),
                            "Precision_Anomaly": report_hdb.get('-1', {}).get('precision', 0),
                        })
                    except Exception as e:
                        print(f"HDBSCAN Failed for {exp_id}: {e}")

    res_df = pd.DataFrame(h3_results)
    res_df.to_csv(os.path.join(RESULTS_DIR, "pca_variance_metrics.csv"), index=False)
    print("\n--- PCA Variance Testing Complete ---")