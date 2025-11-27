import os
import pandas as pd
from statsmodels.stats.multitest import multipletests
import numpy as np

folder_path = "./"
output_file = os.path.join(folder_path, "All_Stats_with_BH.csv")

all_stats = []

for fname in os.listdir(folder_path):
    if fname.endswith(".csv"):
        file_path = os.path.join(folder_path, fname)
        df = pd.read_csv(file_path)
        df["file"] = fname 
        all_stats.append(df)

if not all_stats:
    raise ValueError("No CSV files found in the specified folder!")

df_all = pd.concat(all_stats, ignore_index=True)

df_all["p_value"] = pd.to_numeric(df_all["p_value"], errors='coerce')

df_all["p_value_BH"] = np.nan

for metric in df_all["Metric"].unique():
    idx = df_all["Metric"] == metric
    pvals = df_all.loc[idx, "p_value"].values
    
    if len(pvals) == 0:
        continue
    
    # Apply Benjamini-Hochberg FDR correction
    reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
    df_all.loc[idx, "p_value_BH"] = pvals_corrected

df_all.to_csv(output_file, index=False)
print(f"Saved BH-corrected p-values to: {output_file}")
