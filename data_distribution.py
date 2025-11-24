import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
# POINT THIS TO THE PARENT FOLDER containing both 'Training' and 'Test'
BASE_DIR = r"c:\Users\DiegoPC\Documents\project\FruitClassification\data\fruits-360_100x100\fruits-360"
SUB_DIRS = ["Training", "Test"]

# ==========================================
# 2. DATA LISTS
# ==========================================
GROUPS = {
    "Healthy Apples": [
        "Apple 5", "Apple 10", "Apple 11", "Apple 14", "Apple 17",  
        "Apple 18", "Apple Braeburn 1", "Apple Crimson Snow 1", 
        "Apple Golden 2", "Apple Golden 3", "Apple Granny Smith 1", 
        "Apple Pink Lady 1", "Apple Red 1", "Apple Red 2", "Apple Red 3", 
        "Apple Red Delicious 1", "Apple Red Yellow 1"
    ],
    "Healthy Others": [
        "Peach 1", "Peach 2", "Peach Flat 1",
        "Pear 1", "Pear 2", "Pear 5", "Pear Forrelle 1", "Pear Monster 1", "Pear Red 1", "Pear Williams 1",
        "Tomato 1", "Tomato 2", "Tomato 3", "Tomato 4", "Tomato 5", "tomato 7", "Tomato 10", 
        "Tomato Cherry Maroon 1", "Tomato Cherry Orange 1", "Tomato Cherry Red 1",
        "Tomato Cherry Red 2", "Tomato Cherry Yellow 1", "Tomato Heart 1", "Tomato Yellow 1"
    ],
    "Healthy Bananas": [
        "Banana 1", "Banana 3", "Banana 4", "Banana Lady Finger 1"
    ],
    "Apple Anomalies": [
        "Apple Core 1", "Apple hit 1", "Apple Rotten 1"
    ],
    "Other Anomalies": [
        "Peach 3", "Peach 4", "Peach 5", "Peach 6",
        "Pear 3", "Pear 6", "Pear 7", "Pear 8", "Pear 12",
        "Tomato not Ripen 1"
    ]
}

# ==========================================
# 3. COUNTING LOGIC (Recalculating to be safe)
# ==========================================
def get_count(folder_name):
    total = 0
    valid_exts = {'.jpg', '.jpeg', '.png'}
    for subdir in SUB_DIRS:
        path = os.path.join(BASE_DIR, subdir, folder_name)
        if os.path.exists(path):
            for r, d, f in os.walk(path):
                for file in f:
                    if Path(file).suffix.lower() in valid_exts:
                        total += 1
    return total

def generate_clean_data():
    print(f"Scanning {BASE_DIR}...")
    rows = []
    for category, folders in GROUPS.items():
        print(f"  Processing {category}...")
        
        # FORCE CORRECT TYPE
        if "Anomalies" in category:
            ftype = "Anomaly"
        else:
            ftype = "Healthy"
            
        for folder in folders:
            count = get_count(folder)
            if count > 0:
                rows.append({
                    "Category": category,
                    "Subclass": folder,
                    "Count": count,
                    "Type": ftype
                })
    return pd.DataFrame(rows)

# ==========================================
# 4. PLOTTING LOGIC
# ==========================================
def plot_overall_dashboard(df):
    print("\nGenerating Overall Dashboard...")
    sns.set_style("whitegrid")
    
    # Create Layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    # --- A. PIE CHART (Overall Balance) ---
    ax1 = fig.add_subplot(gs[0, 0])
    type_counts = df.groupby("Type")["Count"].sum()
    
    # Force Colors: Green for Healthy, Red for Anomaly
    colors = ['#66cc66' if idx == 'Healthy' else '#ff6666' for idx in type_counts.index]
    
    ax1.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=colors, explode=[0.05]*len(type_counts), shadow=True,
            textprops={'fontsize': 14, 'weight': 'bold'})
    ax1.set_title("Overall Dataset Balance\n(Healthy vs. Anomalies)", fontsize=16)

    # --- B. BAR CHART (Group Distribution) ---
    ax2 = fig.add_subplot(gs[0, 1])
    cat_counts = df.groupby("Category")["Count"].sum().sort_values(ascending=False)
    
    palette = {
        "Healthy Apples": "forestgreen", "Healthy Others": "lightgreen",
        "Healthy Bananas": "gold", "Apple Anomalies": "darkred", "Other Anomalies": "firebrick"
    }
    
    sns.barplot(x=cat_counts.values, y=cat_counts.index, ax=ax2, palette=palette, edgecolor="black")
    
    # Add labels
    for i, v in enumerate(cat_counts.values):
        ax2.text(v + 100, i, f"{v}", color='black', va='center', fontweight='bold')
    
    ax2.set_title("Total Images per Group", fontsize=16)

    # --- C. APPLE SPECIFIC BAR CHART ---
    ax3 = fig.add_subplot(gs[1, :]) # Bottom row full width
    
    # Filter for Apples
    apple_df = df[df["Category"].str.contains("Apple")]
    if not apple_df.empty:
        h_apples = apple_df[apple_df["Category"] == "Healthy Apples"]["Count"].sum()
        a_apples = apple_df[apple_df["Category"] == "Apple Anomalies"]["Count"].sum()
        
        ax3.barh(["Rotten Apples", "Healthy Apples"], [a_apples, h_apples], 
                 color=["darkred", "forestgreen"], edgecolor="black", height=0.5)
        
        if a_apples > 0:
            ratio = h_apples / a_apples
            ax3.text(h_apples / 2, 1, f"Imbalance Ratio: {ratio:.1f} to 1", 
                     ha='center', va='center', fontsize=20, 
                     bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'))
    
    ax3.set_title("Focus on Apples: Healthy vs. Rotten", fontsize=16)

    plt.suptitle("Fruit Dataset Distribution Report", fontsize=22, weight='bold')
    plt.tight_layout()
    
    save_path = "overall_dataset_balance.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved Graph: {save_path}")

# ==========================================
# 5. RUN
# ==========================================
if __name__ == "__main__":
    df = generate_clean_data()
    if not df.empty:
        plot_overall_dashboard(df)
        df.to_csv("clean_dataset_counts.csv", index=False)
    else:
        print("No images found. Check BASE_DIR.")