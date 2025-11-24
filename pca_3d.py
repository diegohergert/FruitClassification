import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Path to your specific V23 feature file
JOBLIB_FILE = r"paramsv2\feat_Config_v23.joblib"

# We need the list to know which labels are Anomalies
ANOMALIES = [
    "Apple Core 1", "Apple hit 1", "Apple Rotten 1",
    "Peach 3", "Peach 4", "Peach 5", "Peach 6",
    "Pear 3", "Pear 6", "Pear 7", "Pear 8", "Pear 12",
    "Tomato not Ripen 1"
]

def visualize_features():
    # 1. LOAD DATA
    if not os.path.exists(JOBLIB_FILE):
        print(f"Error: File not found at {JOBLIB_FILE}")
        return

    print(f"Loading {JOBLIB_FILE}...")
    df = joblib.load(JOBLIB_FILE)
    
    # Check if empty
    if df.empty:
        print("Dataframe is empty.")
        return

    # 2. PREPARE DATA
    # Convert list of features into a proper 2D Matrix
    X = np.stack(df['features'].values)
    
    # Create Binary Label (Healthy vs Anomaly)
    y = df['label'].apply(lambda x: 'Anomaly' if x in ANOMALIES else 'Healthy').values
    
    print(f"Feature Matrix Shape: {X.shape}")
    print(f"Class Distribution: {pd.Series(y).value_counts().to_dict()}")

    # 3. NORMALIZE (Critical for PCA)
    # Using MinMax because that was your best scaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ==========================================
    # VISUALIZATION 1: PCA 2D PROJECTION
    # ==========================================
    print("Generating PCA Plot...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, 
                    palette={'Healthy': 'green', 'Anomaly': 'red'}, 
                    alpha=0.6, s=15)
    
    plt.title(f"PCA Feature Space Visualization (Config_v23)\nExplained Variance: {pca.explained_variance_ratio_.sum():.2f}", fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    plt.savefig("viz_pca_v23.png", dpi=300)
    print("Saved viz_pca_v23.png")

    # ==========================================
    # VISUALIZATION 2: FEATURE "DNA" COMPARISON
    # ==========================================
    print("Generating Feature Signal Plot...")
    
    # Calculate Mean Vector for Healthy vs Anomaly
    mean_healthy = X_scaled[y == 'Healthy'].mean(axis=0)
    mean_anom = X_scaled[y == 'Anomaly'].mean(axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(mean_healthy, label='Average Healthy Fruit', color='green', linewidth=2, alpha=0.8)
    plt.plot(mean_anom, label='Average Anomaly', color='red', linewidth=2, alpha=0.8)
    
    # Highlighting the diff
    plt.fill_between(range(len(mean_healthy)), mean_healthy, mean_anom, color='gray', alpha=0.1)
    
    plt.title("Average Feature Vector Comparison (Healthy vs. Anomaly)", fontsize=14)
    plt.xlabel("Feature Index (Color -> LBP -> HOG)", fontsize=12)
    plt.ylabel("Normalized Feature Value", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations for feature groups (Approximated based on your extraction order)
    # Stats (12) + Hist (48 or similar) + LBP (~36) + HOG (The rest)
    # You can adjust these text positions based on the graph look
    plt.text(5, 0.9, "Color Stats", fontsize=10, rotation=90)
    plt.text(20, 0.9, "Color Hist", fontsize=10, rotation=90)
    plt.text(50, 0.9, "Texture (LBP)", fontsize=10, rotation=90)
    plt.text(100, 0.9, "Shape (HOG)", fontsize=10, rotation=90)
    
    plt.savefig("viz_feature_dna_v23.png", dpi=300)
    print("Saved viz_feature_dna_v23.png")
    
    plt.show()

if __name__ == "__main__":
    visualize_features()