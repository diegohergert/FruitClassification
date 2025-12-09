import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevent threading errors
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import HDBSCAN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, precision_recall_curve, average_precision_score

#configs and setup

# folders I labeled as healthy apples
APPLES = [
    "Apple 5", "Apple 10", "Apple 11", "Apple 14", "Apple 17",  
    "Apple 18", "Apple Braeburn 1", "Apple Crimson Snow 1", 
    "Apple Golden 2", "Apple Golden 3", "Apple Granny Smith 1", 
    "Apple Pink Lady 1", "Apple Red 1", "Apple Red 2", "Apple Red 3", 
    "Apple Red Delicious 1", "Apple Red Yellow 1"
]
ANOMALIES = ["Apple Core 1", "Apple hit 1", "Apple Rotten 1"] 

OUTPUT_DIR = "h3_comprehensive_analysis_v3/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = [
    # 1. BASELINE: Isolation Forest with PCA
    {
        "id": "Exp1_Baseline",
        "file": "params/feature_set_1.joblib",
        "feat": "hog_hsv_features",
        "scaler": "MinMax",
        "pca": 0.95, 
        "model": "IsolationForest"
    },
    
    # 2. Baseline Edited (Set 7)
    {
        "id": "Exp2_Edited",
        "file": "params/feature_set_1.joblib",
        "feat": "all_features",
        "scaler": "MinMax",
        "pca": 0.95, 
        "model": "IsolationForest"
    },

    # 2. Baseline Edited (Set 7)
    {
        "id": "Exp3_Edited",
        "file": "params/feature_set_1.joblib",
        "feat": "hog_hsv_features",
        "scaler": "MinMax",
        "pca": 0.97, 
        "model": "IsolationForest"
    }, 

    # 2. Baseline Edited (Set 7)
    {
        "id": "Exp4_Edited",
        "file": "params/feature_set_1.joblib",
        "feat": "hog_hsv_features",
        "scaler": "MinMax",
        "pca": 0.90, 
        "model": "IsolationForest"
    },
]

#data loading

def load_and_prep_data(file_path, feature_col, seed=42):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None, None, None

    loaded = joblib.load(file_path)
    data = loaded['data']
    
    # Randomly sample healthy data
    # This ensures each "Round" of CV gets a different subset of healthy apples
    healthy = data[data['label'].isin(APPLES)].sample(min(int(len(data[data['label'].isin(APPLES)])), 20000), random_state=seed)
    
    # Take ALL anomalies (or sample a subset if you want to vary that too)
    anom = data[data['label'].isin(ANOMALIES)].sample(min(int(len(healthy) * .05), len(data[data['label'].isin(ANOMALIES)])), random_state=seed)
    
    df = pd.concat([healthy, anom]).reset_index(drop=True)
    
    # Labels: 1 = Healthy, -1 = Anomaly 
    y_true = df['label'].apply(lambda x: 1 if x in APPLES else -1).values
    
    # Extract Features
    try:
        X = np.array(df[feature_col].tolist())
    except KeyError:
        print(f"Error: Feature column '{feature_col}' not found.")
        return None, None, None
        
    return X, y_true, df

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
    if rows == 0 or cols == 0: return
    
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

#3 visualization functions

def plot_score_histogram(df, exp_name):
    """Plots the distribution of Healthy vs Anomaly scores."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Anomaly_Score', hue='Class', element="step", stat="density", common_norm=False, palette={'Healthy':'green', 'Anomaly':'red'})
    plt.title(f"{exp_name}: Score Distribution\n(Separation = Good, Overlap = Bad)")
    plt.xlabel("Anomaly Score (Higher = More Likely to be Rot)")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{exp_name}_Hist.png"))
    plt.close()

def plot_pr_curve(y_true, scores, exp_name, ap_score):
    """Plots Precision-Recall Curve."""
    # y_true is 1/-1. We need binary 0/1 where 1 is the 'Positive' (Anomaly) class
    y_binary = (y_true == -1).astype(int)
    
    precision, recall, _ = precision_recall_curve(y_binary, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'AP={ap_score:.3f}')
    plt.xlabel('Recall (Percent of Rot Caught)')
    plt.ylabel('Precision (Percent of Alerts that were Real)')
    plt.title(f"{exp_name}: Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{exp_name}_PR_Curve.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, exp_name):
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Rot"])
    plt.figure(figsize=(5, 5))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{exp_name} Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{exp_name}_CM.png"))
    plt.close()

#   main pipeline

def run_experiments():
    print(f"Starting H3 Analysis with Stability Testing. Results will be in: {OUTPUT_DIR}")
    
    #Cross-Validation
    N_ROUNDS = 5  # How many times to repeat the experiment
    SEEDS = [42, 101, 999, 123, 555] # Different random states for each round
    
    raw_results = [] 
    
    for exp in EXPERIMENTS:
        print(f"\n=== Testing Config: {exp['id']} ({N_ROUNDS} Rounds) ===")
        
        experiment_scores = {
            "AUC": [], "AP": [], "F1": [], "Recall": [], "Precision": []
        }

        for i in range(N_ROUNDS):
            current_seed = SEEDS[i]
            print(f"   > Round {i+1}/{N_ROUNDS} (Seed {current_seed})...", end="\r")
            
            # 1. Load & Sample Data
            X, y_true, df = load_and_prep_data(exp['file'], exp['feat'], seed=current_seed)
            if X is None: continue

            # 2. Pipeline (Scaling -> PCA)
            if exp['scaler'] == "MinMax": scaler = MinMaxScaler()
            else: scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            if exp['pca'] is not None:
                pca = PCA(n_components=exp['pca'])
                X = pca.fit_transform(X)

            # 3. Run Model
            n_anom = np.sum(y_true == -1)
            contam = n_anom / len(y_true)
            
            scores = None
            y_pred = None
            
            if exp['model'] == "IsolationForest":
                model = IsolationForest(n_estimators=100, contamination='auto', random_state=current_seed)
                y_pred = model.fit_predict(X)
                scores = -1 * model.decision_function(X) 
            elif exp['model'] == "OCSVM":
                nu_val = min(contam + 0.01, 0.5)
                model = OneClassSVM(kernel="rbf", gamma='auto', nu=nu_val)
                y_pred = model.fit_predict(X)
                scores = -1 * model.decision_function(X)
            else: # HDBSCAN
                model = HDBSCAN(min_cluster_size=10)
                model.fit(X)
                y_pred_raw = model.labels_
                y_pred = np.where(y_pred_raw == -1, -1, 1)
                scores = 1 - model.probabilities_

            # 4. Log Metrics for this Round
            y_binary = (y_true == -1).astype(int)
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            anom_metrics = report.get('-1', {})
            
            experiment_scores["AUC"].append(roc_auc_score(y_binary, scores))
            experiment_scores["AP"].append(average_precision_score(y_binary, scores))
            experiment_scores["F1"].append(anom_metrics.get('f1-score', 0))
            experiment_scores["Recall"].append(anom_metrics.get('recall', 0))
            experiment_scores["Precision"].append(anom_metrics.get('precision', 0))
            
            # Save the graph only for the first round so we don't get 20 images
            if i == 0:
                df['Anomaly_Score'] = scores
                df['Class'] = df['label'].apply(lambda x: 'Healthy' if x in APPLES else 'Anomaly')
                df['Pred_Label'] = y_pred
                plot_score_histogram(df, exp['id'])
                plot_pr_curve(y_true, scores, exp['id'], average_precision_score(y_binary, scores))
                plot_confusion_matrix(y_true, y_pred, exp['id'])
                
                # Save qualitative error examples (only from run 1)
                fn_paths = df[(df['label'].isin(ANOMALIES)) & (df['Pred_Label'] == 1)]['image_path'].tolist()
                if fn_paths: save_image_grid(fn_paths, f"{exp['id']} False Negatives", f"{exp['id']}_FN.png")
                fp_paths = df[(df['label'].isin(APPLES)) & (df['Pred_Label'] == -1)]['image_path'].tolist()
                if fp_paths: save_image_grid(fp_paths, f"{exp['id']} False Positives", f"{exp['id']}_FP.png")

        # 5. Aggregate Results after 5 rounds
        print(f"\n   > Completed {N_ROUNDS} Rounds.")
        
        # Calculate Mean and Std for each metric
        summary = {
            "Experiment": exp['id'],
            "AUC_Mean": np.mean(experiment_scores["AUC"]),
            "AUC_Std": np.std(experiment_scores["AUC"]),
            "F1_Mean": np.mean(experiment_scores["F1"]),
            "F1_Std": np.std(experiment_scores["F1"]),
            "Recall_Mean": np.mean(experiment_scores["Recall"]),
            "AP_Mean": np.mean(experiment_scores["AP"]),
        }
        raw_results.append(summary)

    # final report
    results_df = pd.DataFrame(raw_results)
    csv_path = os.path.join(OUTPUT_DIR, "h3_stability_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    
    print("\n" + "="*60)
    print(f"STABILITY ANALYSIS COMPLETE. CSV SAVED: {csv_path}")
    print("="*60)
    # table print
    print(results_df[['Experiment', 'AUC_Mean', 'AUC_Std', 'F1_Mean']])

if __name__ == "__main__":
    run_experiments()