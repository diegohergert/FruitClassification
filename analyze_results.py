import pandas as pd
import matplotlib.pyplot as plt

# 1. Define the Data
data = {
    "Feature Configuration": ["HOG (Original)", "HOG (PCA)"],
    "Dimensions": [8712, 1267],
    "ARI": [0.9544, 0.9544],
    "Silhouette": [0.1176, 0.1238],
    "Calinski-Harabasz": [516.49, 547.55],
    "Inertia": [838115, 790578]
}

# 2. Create DataFrame
df = pd.DataFrame(data)

def create_clean_table():
    # 3. Setup the Plot
    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.axis('off')

    # 4. Create the Table directly via matplotlib (avoids creating the index column)
    tbl = ax.table(
        cellText=df.values, 
        colLabels=df.columns, 
        loc='center', 
        cellLoc='center',
        colWidths=[0.25, 0.1, 0.1, 0.15, 0.2, 0.15]
    )

    # 5. Styling
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)

    # Style the Cells
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            # Header Styling
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50') # Dark Blue/Grey
            cell.set_height(0.2)
        else:
            # Data Styling
            cell.set_facecolor('#f8f9fa') # Light Grey
            cell.set_height(0.18)

    # 6. Add Title
    plt.title("H1.1 Performance Comparison: Original vs. PCA",
              y=.9, fontsize=12, fontweight='bold')

    # 7. Save
    output_file = 'H1_Best_Metrics_Table_Clean.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Table saved to {output_file}")

if __name__ == "__main__":
    create_clean_table()