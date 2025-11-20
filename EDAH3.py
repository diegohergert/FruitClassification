import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import sys
import cv2
import os
from sklearn.ensemble import IsolationForest
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
Label = {
    "apple_species": [
        "Apple 5", "Apple 6", "Apple 7", "Apple 8", "Apple 9/",
        "Apple 10", "Apple 11", "Apple 12", "Apple 13", "Apple 14",
        "Apple 15", "Apple 16", "Apple 17", "Apple 18", "Apple 19",
        "Apple Braeburn 1", "Apple Crimson Snow 1",
        "Apple Golden 1", "Apple Golden 2", "Apple Golden 3",
        "Apple Granny Smith 1", "Apple Pink Lady 1",
        "Apple Red 1", "Apple Red 2", "Apple Red 3",
        "Apple Red Delicious 1", "Apple Red Yellow 1", "Apple Red Yellow 2"
    ],
    "apple_anomalies": [
        "Apple Core 1", "Apple hit 1", "Apple Rotten 1"
    ]
}

RESULTS_DIR = "results_h3/"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_anomaly_dataset(data, feature_column, contamination=0.05, max_samples=2000):
    """Creates a dataset with mostly healthy apples and a small % of anomalies."""
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
    
    X = np.array(combined_df[feature_column].tolist())
    y_true = combined_df['label'].apply(lambda x: 1 if x in Label["apple_species"] else -1).values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_true, combined_df

def visualize_predictions(y_true, y_pred, df, title_prefix, save_path):
    """Plots Confusion Matrix and Grid of Samples."""
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy (1)", "Anomaly (-1)"])
    
    plt.figure(figsize=(6, 5))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{title_prefix} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path + "_cm.png")
    plt.close()

    # 2. Image Samples
    categories = {
        "TP_Caught_Rot": (y_true == -1) & (y_pred == -1),
        "FN_Missed_Rot": (y_true == -1) & (y_pred == 1),
        "FP_False_Alarm": (y_true == 1) & (y_pred == -1),
        "TN_Healthy": (y_true == 1) & (y_pred == 1)
    }
    
    for cat_name, mask in categories.items():
        indices = np.where(mask)[0]
        if len(indices) == 0: continue

        np.random.shuffle(indices)
        sample_indices = indices[:4]

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        fig.suptitle(f"{cat_name} (Count: {len(indices)})", fontsize=12)
        
        if len(sample_indices) == 1: axes = [axes]

        for i, ax in enumerate(axes):
            if i < len(sample_indices):
                idx = sample_indices[i]
                path = df.iloc[idx]['image_path']
                label = df.iloc[idx]['label']
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img)
                        ax.set_title(label, fontsize=7)
                    else:
                        ax.text(0.5, 0.5, "Img Not Found", ha='center')
                except:
                    ax.text(0.5, 0.5, "Error", ha='center')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}_{cat_name}.png")
        plt.close()

def plot_umap_anomalies(X_data, y_true, y_pred, save_path):
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=y_true,
                        palette={1: 'blue', -1: 'red'},
                        ax=axes[0], s=15, alpha=0.6)
    axes[0].set_title("Ground Truth (Red=Anomaly)")
    
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=y_pred,
                        palette={1: 'blue', -1: 'red'},
                        ax=axes[1], s=15, alpha=0.6)
    axes[1].set_title("Prediction (Red=Anomaly)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    FEATURE_SET_PATH = [
        "params/feature_set_5.joblib", 
        "params/feature_set_6.joblib",
        "params/feature_set_7.joblib",
        "params/feature_set_8.joblib"
    ]
    
    FEAT_COLUMNS = ["hog_lbp_features", "all_features"]
    CONTAMINATION = 0.05 

    h3_results = []

    print("Starting Hypothesis 3 (Anomaly Detection)...")

    for file_path in FEATURE_SET_PATH:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path}...")
            continue
            
        print(f"\nLoading {file_path}...")
        loaded = joblib.load(file_path)
        data = loaded['data']
        
        file_id = os.path.basename(file_path).replace(".joblib", "")

        for feat_col in FEAT_COLUMNS:
            print(f"  Testing {feat_col}...")
            
            X_scaled, y_true, df_subset = create_anomaly_dataset(data, feat_col, contamination=CONTAMINATION)
            
            pca = PCA(n_components=0.95, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            # --- MODEL 1: ISOLATION FOREST ---
            iso = IsolationForest(contamination=CONTAMINATION, random_state=42, n_jobs=-1)
            y_pred_iso = iso.fit_predict(X_pca)
            
            report_iso = classification_report(y_true, y_pred_iso, output_dict=True, zero_division=0)
            
            # Save with Unique ID
            base_name = os.path.join(RESULTS_DIR, f"{file_id}_{feat_col}_ISO")
            plot_umap_anomalies(X_pca, y_true, y_pred_iso, f"{base_name}_UMAP.png")
            visualize_predictions(y_true, y_pred_iso, df_subset, f"ISO ({feat_col})", base_name)
            
            h3_results.append({
                "Model": "Isolation Forest",
                "ParamSet": file_id,
                "Feature": feat_col,
                "F1_Anomaly": report_iso.get('-1', {}).get('f1-score', 0),
                "Recall_Anomaly": report_iso.get('-1', {}).get('recall', 0),
                "Precision_Anomaly": report_iso.get('-1', {}).get('precision', 0),
            })

            # --- MODEL 2: HDBSCAN ---
            hdb = HDBSCAN(min_cluster_size=15, min_samples=5)
            y_pred_hdb_raw = hdb.fit_predict(X_pca)
            y_pred_hdb = np.where(y_pred_hdb_raw == -1, -1, 1)
            
            report_hdb = classification_report(y_true, y_pred_hdb, output_dict=True, zero_division=0)
            
            # Save with Unique ID
            base_name_hdb = os.path.join(RESULTS_DIR, f"{file_id}_{feat_col}_HDB")
            plot_umap_anomalies(X_pca, y_true, y_pred_hdb, f"{base_name_hdb}_UMAP.png")
            visualize_predictions(y_true, y_pred_hdb, df_subset, f"HDB ({feat_col})", base_name_hdb)

            h3_results.append({
                "Model": "HDBSCAN",
                "ParamSet": file_id,
                "Feature": feat_col,
                "F1_Anomaly": report_hdb.get('-1', {}).get('f1-score', 0),
                "Recall_Anomaly": report_hdb.get('-1', {}).get('recall', 0),
                "Precision_Anomaly": report_hdb.get('-1', {}).get('precision', 0),
            })

            print(f"    ISO F1: {report_iso.get('-1', {}).get('f1-score', 0):.3f} | HDB F1: {report_hdb.get('-1', {}).get('f1-score', 0):.3f}")

    res_df = pd.DataFrame(h3_results)
    res_df.to_csv(os.path.join(RESULTS_DIR, "hypothesis_3_metrics.csv"), index=False)
    print("\n--- Hypothesis 3 Complete ---")