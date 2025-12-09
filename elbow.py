import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Configuration
INPUT_FILE = "resultsh2/hypothesis_2_kmeans_elbow_results.csv"  # Replace with your actual CSV filename
OUTPUT_ELBOW_IMG = "elbow_comparison.png"
OUTPUT_SILHOUETTE_IMG = "silhouette_comparison.png"

def plot_elbow_comparison():
    # 2. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}.")
        return
    
    df = pd.read_csv(INPUT_FILE)

    # 3. Identify the "Best" Feature Set
    # We find the single row with the highest Silhouette Score to determine which Feature Set to plot.
    best_idx = df['silhouette_score'].idxmax()
    best_row = df.loc[best_idx]
    
    # We lock onto this specific file and column (e.g., "feature_set_1.joblib" and "hsv_features")
    target_file = best_row['feature_set_file']
    target_col = best_row['feature_column']
    
    print(f"--- Best Configuration Found ---")
    print(f"Set: {target_file}")
    print(f"Features: {target_col}")
    print(f"Max Score: {best_row['silhouette_score']:.4f}")
    
    # 4. Filter Data for both types (Original & PCA)
    # We want all rows that match the file and column, regardless of 'feature_type' or 'k'
    mask = (df['feature_set_file'] == target_file) & (df['feature_column'] == target_col)
    plot_data = df[mask].copy()
    
    # Sort by K so lines draw correctly
    plot_data.sort_values(by='k', inplace=True)

    # 5. Plot 1: Elbow Comparison (Inertia)
    plt.figure(figsize=(10, 6))
    
    # Draw line for Original
    orig_data = plot_data[plot_data['feature_type'] == 'Original']
    plt.plot(orig_data['k'], orig_data['inertia'], 
             marker='o', linestyle='-', color='blue', label='Original (No PCA)')
    
    # Draw line for PCA
    pca_data = plot_data[plot_data['feature_type'] == 'PCA']
    plt.plot(pca_data['k'], pca_data['inertia'], 
             marker='s', linestyle='--', color='red', label='With PCA (95%)')
    
    plt.title(f"Elbow Method Comparison: Inertia vs K\n({target_col})")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_ELBOW_IMG)
    print(f"Saved Elbow Comparison to {OUTPUT_ELBOW_IMG}")
    plt.close()

    # 6. Plot 2: Silhouette Comparison
    plt.figure(figsize=(10, 6))
    
    plt.plot(orig_data['k'], orig_data['silhouette_score'], 
             marker='o', linestyle='-', color='green', label='Original (No PCA)')
    
    plt.plot(pca_data['k'], pca_data['silhouette_score'], 
             marker='s', linestyle='--', color='orange', label='With PCA (95%)')
    
    plt.title(f"Cluster Quality Comparison: Silhouette Score vs K\n({target_col})")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_SILHOUETTE_IMG)
    print(f"Saved Silhouette Comparison to {OUTPUT_SILHOUETTE_IMG}")
    plt.close()

if __name__ == "__main__":
    plot_elbow_comparison()