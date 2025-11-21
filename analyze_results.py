import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

# --- CONFIGURATION ---
RESULTS_DIR = "results/"          # H1
RESULTS_DIR_H2 = "resultsh2/"     # H2
RESULTS_DIR_H3 = "results_h3_pca_test/" # H3
OUTPUT_DIR = "comprehensive_audit/" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="white")

def get_clean_id(df, file_col='feature_set_file', feat_col='feature_column'):
    """
    Creates a unique ID like 'Set5 | hog_hsv' for the Y-axis
    """
    # Extract just "set_5" from "params/feature_set_5.joblib"
    df['Set_ID'] = df[file_col].astype(str).apply(lambda x: re.search(r'feature_set_(\d+)', x).group(0) if re.search(r'feature_set_(\d+)', x) else 'Unknown')
    df['Short_Feat'] = df[feat_col].astype(str).str.replace('_features', '')
    df['Unique_ID'] = df['Set_ID'] + " | " + df['Short_Feat']
    return df

def analyze_h1_comprehensive():
    print("Generating H1 Audit (Apples vs Bananas)...")
    path = os.path.join(RESULTS_DIR, "hypothesis_1_results.csv")
    if not os.path.exists(path): return

    df = pd.read_csv(path)
    df = get_clean_id(df)

    # Pivot: Rows=Features/Params, Cols=Preprocessing(PCA/Original)
    heatmap_data = df.pivot_table(index='Unique_ID', columns='feature_type', values='ari')
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn", linewidths=.5)
    plt.title("H1: Feature Set & Param Audit (ARI Score)\n(Which inputs fail to separate Apples/Bananas?)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "H1_Audit_Heatmap.png"))
    plt.close()

def analyze_h2_comprehensive():
    print("Generating H2 Audit (Species Clustering)...")
    
    # Load both K-Means and HDBSCAN
    kmeans_path = os.path.join(RESULTS_DIR_H2, "hypothesis_2_kmeans_elbow_results.csv")
    hdbscan_path = os.path.join(RESULTS_DIR_H2, "hypothesis_2_hdbscan_results.csv")
    
    if not os.path.exists(kmeans_path) or not os.path.exists(hdbscan_path): return

    # Process K-Means: Get the MAX ARI achievable for each feature set (Best K)
    df_k = pd.read_csv(kmeans_path)
    # Note: K-Means file might not have ARI if you didn't calculate it for every K, 
    # but based on your H2 script, you only calculated Inertia/Silhouette. 
    # If ARI is missing, we use Silhouette.
    metric = 'silhouette_score' # Fallback
    
    df_k = get_clean_id(df_k)
    # Find best score for each Unique ID
    best_k = df_k.groupby('Unique_ID')[metric].max().reset_index()
    best_k['Model'] = 'Best K-Means'
    
    # Process HDBSCAN
    df_h = pd.read_csv(hdbscan_path)
    df_h = get_clean_id(df_h)
    best_h = df_h[['Unique_ID', metric]].copy() # HDBSCAN only has one run per feat
    best_h['Model'] = 'HDBSCAN'
    
    # Combine
    combined = pd.concat([best_k, best_h])
    
    # Pivot
    heatmap_data = combined.pivot_table(index='Unique_ID', columns='Model', values=metric)
    
    plt.figure(figsize=(8, 12))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
    plt.title(f"H2: Algorithm Comparison ({metric})\n(Does density-based clustering beat centroid-based?)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "H2_Audit_Heatmap.png"))
    plt.close()

def analyze_h3_comprehensive():
    print("Generating H3 Audit (Anomaly Detection)...")
    path = os.path.join(RESULTS_DIR_H3, "pca_variance_metrics.csv")
    if not os.path.exists(path): return

    df = pd.read_csv(path)
    # H3 CSV uses different column names ('ParamSet', 'Feature')
    df['Short_Feat'] = df['Feature'].str.replace('_features', '')
    df['Unique_ID'] = df['ParamSet'] + " | " + df['Short_Feat']
    
    # Create "Pipeline Config" column: Scaler + PCA
    df['Pipeline'] = df['Scaler'] + " | PCA=" + df['PCA_Var'].astype(str)

    models = df['Model'].unique()

    for model in models:
        subset = df[df['Model'] == model]
        
        # PIVOT: Row=Param/Feature, Col=Pipeline(Scaler+PCA)
        heatmap_data = subset.pivot_table(index='Unique_ID', columns='Pipeline', values='F1_Anomaly')
        
        # Sort rows by average performance to put best features at top
        heatmap_data['mean'] = heatmap_data.mean(axis=1)
        heatmap_data = heatmap_data.sort_values('mean', ascending=False)
        heatmap_data = heatmap_data.drop(columns='mean')

        plt.figure(figsize=(12, 10)) 
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=.5)
        plt.title(f"H3: {model} Configuration Audit (F1 Score)\nRows=Inputs, Cols=Pipeline Settings")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"H3_Audit_{model}.png"))
        plt.close()

if __name__ == "__main__":
    analyze_h1_comprehensive()
    analyze_h2_comprehensive()
    analyze_h3_comprehensive()
    print(f"\nAudit Complete. Open folder '{OUTPUT_DIR}' to decide what to cut.")