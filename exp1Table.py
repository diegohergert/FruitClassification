# create_h1_2_feature_ablation.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_PATH = "resultsh2/hypothesis_2_hdbscan_results.csv"

def plot_ablation():
    if not os.path.exists(CSV_PATH): return
    df = pd.read_csv(CSV_PATH)
    
    # Clean Feature Names
    df['Features'] = df['feature_column'].apply(lambda x: x.replace('_features', '').replace('_', '+').upper())
    
    # Group by Feature Type and calculate Mean ARI
    df_grouped = df.groupby('Features')['ari'].mean().reset_index().sort_values('ari', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_grouped, x='ari', y='Features', palette='viridis')
    
    plt.title("Feature Ablation: Which features drive Apple Separation?", fontsize=14)
    plt.xlabel("Average ARI Score", fontsize=12)
    plt.ylabel("Feature Combination", fontsize=12)
    plt.axvline(x=0, color='black', linewidth=1)
    
    plt.savefig("H1_2_Feature_Ablation.png", dpi=300, bbox_inches='tight')
    print("Saved H1_2_Feature_Ablation.png")

if __name__ == "__main__":
    plot_ablation()