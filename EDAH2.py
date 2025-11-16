import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import sys
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler
import os

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
        "Apple Core 1", "Apple hit 1", "Apple Rotten 1", "Apple worm 1"
    ],
    "banana": [
        "Banana 1", "Banana 3", "Banana 4", "Banana Lady Finger 1"
    ]
}

def splitData(data, feature_column):
    task_df = data[data['label'].isin(Label["apple_species"])].copy()

    X = np.array(task_df[feature_column].tolist())
    y = task_df['label']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

if __name__ == "__main__":
    FEATURE_SET_PATH = [
        "params/feature_set_1.joblib",
        "params/feature_set_2.joblib",
        "params/feature_set_3.joblib",
        "params/feature_set_4.joblib",
        "params/feature_set_5.joblib",
        "params/feature_set_6.joblib",
        "params/feature_set_7.joblib",
        "params/feature_set_8.joblib"
    ]

    FEAT_COLUMNS = [
        "hsv_features",
        "lbp_features",
        "hog_features",
        "hog_lbp_features",
        "hog_hsv_features",
        "lbp_hsv_features",
        "all_features"
    ]

    RESULTS_DIR = "resultsh2/"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    kmeans_results = []
    hdbscan_results = []

    # H2. Discovering natural groupings among apple species
    print("Starting Hypothesis 2 Experiments...")
    for file_path in FEATURE_SET_PATH:
        if not os.path.exists(file_path):
            print(f"Feature set file {file_path} not found. Skipping.", file=sys.stderr)
            continue
        print(f"Loading feature set from {file_path}...")
        try:
            loaded = joblib.load(file_path)
            data = loaded['data']
            param_config = loaded['parameters']
        except Exception as e:
            print(f"Error loading {file_path}: {e}", file=sys.stderr)
            continue

        df = data[data['label'].isin(Label["apple_species"])].copy()
        del data, loaded
        print("loaded data shape:", df.shape)

        for feat_col in FEAT_COLUMNS:
            print("Testing feature:", feat_col)
            X_scaled, y_labels = splitData(df, feat_col)

            pca = PCA(n_components=0.95, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            print(f"PCA reduced dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]} components.")

            feature_sets = [
                ("Original", X_scaled),
                ("PCA", X_pca)
            ]

            for feat_name, X_data in feature_sets:
                print("Running analysis on feature set:", feat_name)

                #UMAP visualization
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                X_umap = reducer.fit_transform(X_data)
                plot_df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
                plot_df['label'] = y_labels.values
                plt.figure(figsize=(12, 10))
                sns.scatterplot(data=plot_df, x='UMAP1', y='UMAP2', hue='label', s=10, alpha=0.7)
                plt.title(f"UMAP Projection ({feat_name} Features)")
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                file_name = file_path.split('/')[-1].replace('.joblib', f'_{feat_col}_{feat_name}_umap.png')
                plt.savefig(os.path.join(RESULTS_DIR, file_name), bbox_inches='tight')
                plt.close()

                #KMeans clustering
                for k in range (2, 30, 2):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    y_pred = kmeans.fit_predict(X_data)
                    kmeans_results.append({
                        'feature_set_file': file_path,
                        'feature_column': feat_col,
                        'feature_type': feat_name,
                        'task': 'H2-KMeans-Elbow',
                        'k': k,
                        'silhouette_score': silhouette_score(X_data, y_pred),
                        'calinski_harabasz_score': calinski_harabasz_score(X_data, y_pred),
                        'inertia': kmeans.inertia_
                    })

                #HDBSCAN clustering
                hdbscan = HDBSCAN(min_cluster_size=12, gen_min_span_tree=True)
                y_pred = hdbscan.fit_predict(X_data)
                n_clusters_found = len(set(y_pred)) - (1 if -1 in y_pred else 0)
                n_noise_points = np.sum(y_pred == -1)

                if n_clusters_found > 1:
                    non_noise_mask = (y_pred != -1)
                    X_data_filtered = X_data[non_noise_mask]
                    y_pred_filtered = y_pred[non_noise_mask]
                    y_labels_filtered = y_labels.iloc[non_noise_mask]
                    
                    sil_score = silhouette_score(X_data_filtered, y_pred_filtered)
                    ari = adjusted_rand_score(y_labels_filtered, y_pred_filtered)
                    ch_score = calinski_harabasz_score(X_data_filtered, y_pred_filtered)
                else:
                    sil_score, ari, ch_score = None, None, None
                
                hdbscan_results.append({
                    'feature_set_file': file_path,
                    'feature_column': feat_col,
                    'feature_type': feat_name,
                    'param_config': param_config,
                    'n_dimensions': X_data.shape[1],
                    'task': 'H2-HDBSCAN',
                    'n_clusters_found': n_clusters_found,
                    'n_noise_points': n_noise_points,
                    'ari': ari,
                    'silhouette_score': sil_score,
                    'calinski_harabasz_score': ch_score
                })

                print(f"Completed analysis for {feat_name} features from {feat_col} in {file_path}.")
    results_kmeans_df = pd.DataFrame(kmeans_results)
    results_kmeans_df.to_csv(os.path.join(RESULTS_DIR, "hypothesis_2_kmeans_elbow_results.csv"), index=False)
    
    results_hdbscan_df = pd.DataFrame(hdbscan_results)
    results_hdbscan_df = results_hdbscan_df.sort_values(by="ari", ascending=False)
    results_hdbscan_df.to_csv(os.path.join(RESULTS_DIR, "hypothesis_2_hdbscan_results.csv"), index=False)

    print("\n--- HYPOTHESIS 2 EXPERIMENTS COMPLETE ---")
    print("Saved K-Means elbow results to 'hypothesis_2_kmeans_elbow_results.csv'")
    print("Saved HDBSCAN results to 'hypothesis_2_hdbscan_results.csv'")
    print("\nTop 5 HDBSCAN ARI results (how well discovered clusters match true species):")
    print(results_hdbscan_df.head(5))