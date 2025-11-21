import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# =============================================================================
#   USER CONFIGURATION: ENTER YOUR TOP 2 CHOICES HERE
# =============================================================================

H1_CONFIGS = [
    # Choice 1 (The Best)
    {"name": "H1_Best", "file": "params/feature_set_4.joblib", "feat": "hog_hsv_features"},
    # Choice 2 (The Runner Up)
    {"name": "H1_Alt",  "file": "params/feature_set_8.joblib", "feat": "hog_hsv_features"},
]

# HYPOTHESIS 2 (Species Clustering)
H2_CONFIGS = [
    # Choice 1
    {"name": "H2_Best", "file": "params/feature_set_4.joblib", "feat": "hsv_features", "model": "KMeans"},
    # Choice 2
    {"name": "H2_Alt",  "file": "params/feature_set_7.joblib", "feat": "all_features", "model": "HDBSCAN"},
]

# HYPOTHESIS 3 (Anomaly Detection)
H3_CONFIGS = [
    # Choice 1
    {"name": "H3_Best", "file": "params/feature_set_1.joblib", "feat": "hog_hsv_features", "scaler": "MinMax", "pca": 0.95, "model": "IsolationForest"},
    # Choice 2
    {"name": "H3_Alt",  "file": "params/feature_set_7.joblib", "feat": "hog_hsv_features", "scaler": "MinMax", "pca": 0.99, "model": "HDBSCAN"},
]

# =============================================================================
#   GLOBAL SETUP
# =============================================================================

OUTPUT_DIR = "final_report_images/error_analysis_top2_adjusted/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

APPLES = [
    "Apple 5", "Apple 10", 
    "Apple 11", "Apple 12", "Apple 14", "Apple 15", "Apple 16", 
     "Apple 18", "Apple 19", "Apple Braeburn 1", "Apple Crimson Snow 1", 
     "Apple Golden 2", "Apple Golden 3", "Apple Granny Smith 1", 
    "Apple Pink Lady 1", "Apple Red 1", "Apple Red 2", "Apple Red 3", 
    "Apple Red Delicious 1", "Apple Red Yellow 1", "Apple Red Yellow 2"
]
BANANAS = ["Banana 1", "Banana 3", "Banana 4", "Banana Lady Finger 1"]
#"Apple Golden 1", and "Apple 9" are a bit iffy but kept for now
ANOMALIES = ["Apple Core 1", "Apple hit 1", "Apple Rotten 1"] #"Apple 7" "Apple 8" "Apple 13" "Apple 17" are half healthy half not so really hard to classify 

sns.set_theme(style="white")

def save_image_grid(image_paths, title, filename, n=16):
    images = []
    count = 0
    for p in image_paths:
        if count >= n: break
        img = cv2.imread(p)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            count += 1
    
    if not images: return

    rows = int(np.ceil(np.sqrt(len(images))))
    cols = int(np.ceil(len(images) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle(title, fontsize=14)
    
    if rows == 1 and cols == 1: axes = [axes]
    else: axes = axes.flat

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  -> Saved: {filename}")

# =============================================================================
#   HYPOTHESIS 1 ANALYSIS
# =============================================================================
def analyze_h1(config):
    name = config['name']
    print(f"\nProcessing H1 Config: {name}...")
    
    if not os.path.exists(config['file']): return

    loaded = joblib.load(config['file'])
    data = loaded['data']
    df = data[data['label'].isin(APPLES + BANANAS)].copy()
    df['True_Class'] = df['label'].apply(lambda x: 0 if x in APPLES else 1)
    
    X = np.array(df[config['feat']].tolist())
    X = MinMaxScaler().fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_pca)
    
    if accuracy_score(df['True_Class'], y_pred) < 0.5:
        y_pred = 1 - y_pred
    df['Predicted_Class'] = y_pred
    
    # 1. Confusion Matrix
    cm = confusion_matrix(df['True_Class'], y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Apple', 'Banana'])
    plt.figure(figsize=(5, 4))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"H1 ({name}): Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_Confusion_Matrix.png"))
    plt.close()
    
    # 2. Misclassified Grid
    errors = df[df['True_Class'] != df['Predicted_Class']]
    if len(errors) > 0:
        save_image_grid(errors['image_path'].tolist(), f"{name}: Misclassified Fruits", f"{name}_Failures.png")

    # 3. PC1 Separation
    df['PC1'] = X_pca[:, 0]
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='PC1', hue='True_Class', element="step", stat="density", common_norm=False, palette={0:'red', 1:'yellow'})
    plt.title(f"H1 ({name}): PC1 Separation")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_PC1_Separation.png"))
    plt.close()

# =============================================================================
#   HYPOTHESIS 2 ANALYSIS
# =============================================================================
def analyze_h2(config):
    name = config['name']
    print(f"\nProcessing H2 Config: {name}...")
    
    if not os.path.exists(config['file']): return

    loaded = joblib.load(config['file'])
    data = loaded['data']
    df = data[data['label'].isin(APPLES)].copy()
    X = np.array(df[config['feat']].tolist())
    
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=0.95).fit_transform(X)
    
    if config['model'] == "HDBSCAN":
        model = HDBSCAN(min_cluster_size=10)
    else:
        model = KMeans(n_clusters=2, random_state=42)
    
    labels = model.fit_predict(X)
    df['Cluster'] = labels
    
    # Find Messiest Cluster
    messiest_cluster = -1
    max_unique = 0
    for c in np.unique(labels):
        if c == -1: continue
        subset = df[df['Cluster'] == c]
        unique_species = subset['label'].nunique()
        if unique_species > max_unique and len(subset) > 10:
            max_unique = unique_species
            messiest_cluster = c
            
    if messiest_cluster != -1:
        paths = df[df['Cluster'] == messiest_cluster]['image_path'].tolist()
        save_image_grid(paths, f"{name}: Cluster {messiest_cluster} (Mixed Species)", f"{name}_Confused_Cluster.png")

# =============================================================================
#   HYPOTHESIS 3 ANALYSIS
# =============================================================================
def analyze_h3(config):
    name = config['name']
    print(f"\nProcessing H3 Config: {name}...")
    
    if not os.path.exists(config['file']): return

    loaded = joblib.load(config['file'])
    data = loaded['data']
    
    healthy = data[data['label'].isin(APPLES)].sample(4000, random_state=42)
    anom = data[data['label'].isin(ANOMALIES)]
    df = pd.concat([healthy, anom]).reset_index(drop=True)
    y_true = df['label'].apply(lambda x: 1 if x in APPLES else -1).values
    
    X = np.array(df[config['feat']].tolist())
    
    if config['scaler'] == "MinMax": scaler = MinMaxScaler()
    else: scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X = PCA(n_components=config['pca']).fit_transform(X)
    
    if config['model'] == "IsolationForest":
        contam = len(anom) / len(df)
        model = IsolationForest(contamination=contam, random_state=42)
        y_pred = model.fit_predict(X)
    else:
        hdb = HDBSCAN(min_cluster_size=5)
        y_pred_raw = hdb.fit_predict(X)
        y_pred = np.where(y_pred_raw == -1, -1, 1)
        
    df['Predicted'] = y_pred
    df['True'] = y_true
    
    # Errors
    fn_paths = df[(df['True'] == -1) & (df['Predicted'] == 1)]['image_path'].tolist()
    save_image_grid(fn_paths, f"{name}: False Negatives (Missed Rot)", f"{name}_False_Negatives.png")
    
    fp_paths = df[(df['True'] == 1) & (df['Predicted'] == -1)]['image_path'].tolist()
    save_image_grid(fp_paths, f"{name}: False Positives (False Alarm)", f"{name}_False_Positives.png")

def plot_anomaly_scores(config):
    print(f"\nGenerating Anomaly Score Distribution for {config['name']}...")
    if not os.path.exists(config['file']): return

    loaded = joblib.load(config['file'])
    data = loaded['data']
    
    # Prepare Data
    healthy = data[data['label'].isin(APPLES)].sample(2000, random_state=42)
    anom = data[data['label'].isin(ANOMALIES)]
    df = pd.concat([healthy, anom]).reset_index(drop=True)
    
    # 0 = Healthy, 1 = Anomaly (for plotting purposes)
    df['Type'] = df['label'].apply(lambda x: 'Healthy' if x in APPLES else 'Anomaly')
    
    X = np.array(df[config['feat']].tolist())
    
    if config['scaler'] == "MinMax": scaler = MinMaxScaler()
    else: scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = PCA(n_components=config['pca']).fit_transform(X)
    
    # Get Scores
    if config['model'] == "IsolationForest":
        model = IsolationForest(contamination=len(anom)/len(df), random_state=42)
        model.fit(X)
        # decision_function: Lower = more abnormal. We multiply by -1 so Higher = More Abnormal
        scores = -1 * model.decision_function(X) 
    else:
        # HDBSCAN outlier_scores_
        import hdbscan
        model = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
        model.fit(X)
        scores = model.outlier_scores_

    df['Anomaly_Score'] = scores

    # PLOT
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Anomaly_Score', hue='Type', element="step", stat="density", common_norm=False, palette={'Healthy':'green', 'Anomaly':'red'})
    plt.title(f"Distribution of Anomaly Scores ({config['name']})\n(Less Overlap = Better Separation)")
    plt.xlabel("Anomaly Score (Higher = More likely to be Rot)")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{config['name']}_Score_Distribution.png"))
    plt.close()
# =============================================================================
#   MAIN LOOP
# =============================================================================
if __name__ == "__main__":
    print("Starting Top 2 Error Analysis...")
    
    for cfg in H1_CONFIGS: analyze_h1(cfg)
    for cfg in H2_CONFIGS: analyze_h2(cfg)
    for cfg in H3_CONFIGS: analyze_h3(cfg)
    for cfg in H3_CONFIGS: plot_anomaly_scores(cfg)
    
    print(f"\nDone! Images saved in '{OUTPUT_DIR}'")