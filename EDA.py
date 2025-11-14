from sklearnex import patch_sklearn
patch_sklearn()
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import sys
from sklearn.cluster import KMeans
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

def overallLabel(label):
    if label in Label["apple_species"]:
        return "apple"
    elif label in Label["banana"]:
        return "banana"
    return None

def splitData(data, feature_column):
    data['Fruit_Type'] = data['label'].apply(overallLabel)
    task_df = data.dropna(subset=['Fruit_Type']).copy()

    counts = task_df['Fruit_Type'].value_counts()
    min_class_size = counts.min()
    
    print(f"Original counts:\n{counts}")
    print(f"Balancing to {min_class_size} samples per class.")

    balanced_df = pd.DataFrame()
    for label_name in counts.index:
        class_subset = task_df[task_df['Fruit_Type'] == label_name].sample(
            min_class_size, 
            random_state=42 # for reproducibility
        )
        balanced_df = pd.concat([balanced_df, class_subset])
    
    print(f"Balanced dataset shape: {balanced_df.shape}")

    X = np.array(balanced_df[feature_column].tolist())
    y = balanced_df['Fruit_Type']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

if __name__ == "__main__":
    FEATURE_SET_PATH = [
        "params/feature_set_1.joblib",
        "params/feature_set_2.joblib",
        "params/feature_set_3.joblib",
        "params/feature_set_4.joblib"
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

    RESULTS_DIR = "results/"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    master_results = []

    # H1. clustering algorithm can effectively seperate apples and bananas
    print("Starting Hypothesis 1 Experiments...")
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

        labels_keep = Label["apple_species"] + Label["banana"]
        df = data[data['label'].isin(labels_keep)].copy()
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
                plt.figure(figsize=(10, 8))
                sns.scatterplot(data=plot_df, x='UMAP1', y='UMAP2', hue='label', s=10, alpha=0.7)
                plt.title(f"UMAP Projection ({feat_name} Features)")
                plt.legend(loc='best')
                file_name = file_path.split('/')[-1].replace('.joblib', f'_{feat_col}_{feat_name}_umap.png')
                plt.savefig(os.path.join(RESULTS_DIR, file_name))
                plt.close()

                #KMeans clustering
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                y_pred = kmeans.fit_predict(X_data)

                #evaluation metrics
                sil_score = silhouette_score(X_data, y_pred)
                ari = adjusted_rand_score(y_labels.map({'apple':0, 'banana':1}), y_pred)
                ch_score = calinski_harabasz_score(X_data, y_pred)
                inertia = kmeans.inertia_

                print(f"{feat_name} -> ARI: {ari:.4f}, Silhouette: {sil_score:.4f}")

                master_results.append({
                    'feature_set_file': file_path,
                    'feature_column': feat_col,
                    'feature_type': feat_name,
                    'param_config': param_config,
                    'n_dimensions': X_data.shape[1],
                    'task': 'H1',
                    'ari': ari,
                    'silhouette_score': sil_score,
                    'calinski_harabasz_score': ch_score,
                    'inertia': inertia
                })

                print(f"Completed analysis for {feat_name} features from {feat_col} in {file_path}.")
    results_df = pd.DataFrame(master_results)
    results_df = results_df.sort_values(by="ari", ascending=False)
    results_df.to_csv(os.path.join(RESULTS_DIR, "hypothesis_1_results.csv"), index=False)

    print("Top 10 ari results:")
    print(results_df.head(10))