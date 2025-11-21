import pandas as pd
import os

# --- Configuration ---
# Update these paths to match your recent experiment output
RESULTS_DIR = "results_h3_pca_test/"
INPUT_FILE = os.path.join(RESULTS_DIR, "pca_variance_metrics.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "pca_variance_metrics_SORTED.csv")

SORT_BY_COLUMN = "F1_Anomaly" # Sorting by F1 Score
TOP_N_TO_PRINT = 15           # How many top results to show
# ---------------------

def sort_results(file_path, output_path, sort_by):
    """
    Loads the results CSV, sorts it by the specified column,
    prints the top results, and saves the sorted file.
    """
    
    # 1. Check if the input file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        # Fallback: Check if file is in the current directory
        if os.path.exists(os.path.basename(file_path)):
            file_path = os.path.basename(file_path)
            print(f"Found file in current directory. Using: {file_path}")
        else:
            print("Please make sure the results file exists.")
            return

    # 2. Read the CSV file into a DataFrame
    print(f"Loading results from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
        
    # 3. Clean column names (remove accidental spaces) and check existence
    df.columns = df.columns.str.strip()
    
    if sort_by not in df.columns:
        print(f"Error: Column '{sort_by}' not found in the CSV.")
        print(f"Available columns are: {list(df.columns)}")
        return

    # 4. Sort the DataFrame (descending order = Highest Score first)
    df_sorted = df.sort_values(by=sort_by, ascending=False)
    
    # 5. Print the top N results to the console
    print(f"\n--- Top {TOP_N_TO_PRINT} Results by {sort_by} ---")
    # Using to_string() ensures columns aren't hidden in the print output
    print(df_sorted.head(TOP_N_TO_PRINT).to_string(index=False))
    
    # 6. Save the *entire* sorted DataFrame to a new file
    try:
        df_sorted.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved fully sorted results to: {output_path}")
    except Exception as e:
        print(f"Error saving new CSV file: {e}")

if __name__ == "__main__":
    sort_results(INPUT_FILE, OUTPUT_FILE, SORT_BY_COLUMN)