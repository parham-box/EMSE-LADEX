import os   
import csv
from collections import defaultdict
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
import re

# --- Configuration ---
APPROACHES = [
    "Public-LADEX-LLM-NA",
    "Public-LADEX-LLM-LLM"
]
APPROACH_A = APPROACHES[0]
APPROACH_B = APPROACHES[1]

MODELS = ["O4Mini","gpt41mini","local"]
RUNS = range(1, 6)
THRESHOLDS_ORDER = ["no"]
METRIC_TYPES = ["Completeness", "Correctness"]

METHOD_CONFIG = {
    "Completeness": {
        "METHOD": "gt",
        "DIRECTION": "GT_vs_GEN"
    },
    "Correctness": {
        "METHOD": "gen",
        "DIRECTION": "GEN_vs_GT"
    }
}

# --- Data Storage ---
def nested_dict():
    return defaultdict(nested_dict)

avg_data = nested_dict()
raw_data_by_id = nested_dict()
methods_by_direction = defaultdict(set)

# --- 1. Data Extraction and Parsing ---
print("Starting data extraction...")
for approach in APPROACHES:
    print(f"  Processing Approach: {approach}")
    for model_base in MODELS:
        for run in RUNS:
            model_run_prefix = f"{model_base}-{run}"
            base_dir = f"../../Results/{approach}/{model_run_prefix}"
            results_dir = os.path.join(base_dir, "results")

            if not os.path.isdir(results_dir): continue
            subdir_path = results_dir
            for fname in os.listdir(subdir_path):
                if not fname.endswith(".csv"): continue
                # Check if file matches either method
                if not any(METHOD_CONFIG[metric]["METHOD"] in fname for metric in METRIC_TYPES): continue
                is_gt_vs_gen = "_gt" in fname
                direction = "GT_vs_GEN" if is_gt_vs_gen else "GEN_vs_GT"
                # Determine which metric this file corresponds to
                for metric_type, config in METHOD_CONFIG.items():
                    if config["METHOD"] in fname and config["DIRECTION"] == direction:
                        methods_by_direction[direction].add(config["METHOD"])
                        file_path = os.path.join(subdir_path, fname)
                        with open(file_path, "r", encoding="utf-8") as f:
                            reader = csv.reader(f)
                            try:
                                header = next(reader)
                            except StopIteration: continue

                            column_map = {}
                            for i, col_name in enumerate(header):
                                match = re.search(r'\((thr\s(.*?))\)', col_name)
                                if not match: continue
                                threshold = match.group(2)
                                metric_name = ""
                                if "Coverage1" in col_name:
                                    metric_name = "Completeness" if is_gt_vs_gen else "Correctness"
                                elif "Coverage2" in col_name:
                                    metric_name = "Correctness" if is_gt_vs_gen else "Completeness"
                                if metric_name == metric_type:  # Only store data for the matching metric
                                    column_map[i] = {'metric': metric_name, 'threshold': threshold}

                            for row in reader:
                                if not row or not row[0]: continue

                                # Store data for Tests 1 & 2
                                if row[0].startswith("Average thr"):
                                    parts = row[0].split(",") + row[1:]
                                    threshold = parts[0].replace("Average thr ", "")
                                    C1, C2 = float(parts[1]), float(parts[2])
                                    comp, corr = (C1, C2) if is_gt_vs_gen else (C2, C1)
                                    if metric_type == "Completeness":
                                        avg_data[direction]["Completeness"][config["METHOD"]][threshold][model_base][approach][run] = comp
                                    elif metric_type == "Correctness":
                                        avg_data[direction]["Correctness"][config["METHOD"]][threshold][model_base][approach][run] = corr
                                # Store data for Tests 3 & 4
                                elif row[0].replace('.', '', 1).isdigit():
                                    file_id = row[0]
                                    for col_idx, mapping in column_map.items():
                                        try:
                                            value = float(row[col_idx])
                                            metric = mapping['metric']
                                            threshold = mapping['threshold']
                                            if metric == metric_type:
                                                raw_data_by_id[direction][metric][config["METHOD"]][threshold][model_base][approach][run][file_id] = value
                                        except (ValueError, IndexError): continue

print("Data extraction complete.")
print("-" * 30)

# --- A12 Computation Function ---
def compute_a12(our, base):
    if not our or not base: return 0
    count = ties = 0
    for o in our:
        for b in base:
            if o > b: count += 1
            elif o == b: ties += 1
    total = len(our) * len(base)
    return (count + 0.5 * ties) / total if total > 0 else 0

# --- Rounding Function ---
def sround(value):
    if isinstance(value, str):  # Handle 'N/A' case
        return value
    if abs(value) < 0.01:
        # Convert to string to find first non-zero digit
        str_val = f"{value:.10f}".rstrip('0')
        decimal_part = str_val.split('.')[1] if '.' in str_val else ''
        for i, digit in enumerate(decimal_part):
            if digit != '0':
                # Round to the position of the first non-zero digit
                return round(value, i + 1)
        return value  # Return as is if no non-zero digit found
    return round(value, 2)

# --- Statistical Analysis Function ---
def perform_statistical_analysis(output_filename):
    stat_rows = []
    for metric_type in METRIC_TYPES:
        METHOD = METHOD_CONFIG[metric_type]["METHOD"]
        DIRECTION = METHOD_CONFIG[metric_type]["DIRECTION"]
        for threshold in THRESHOLDS_ORDER:
            our_vals, base_vals = [], []
            for model in MODELS:
                for run in RUNS:
                    our_val = avg_data[DIRECTION][metric_type][METHOD][threshold][model][APPROACH_A].get(run)
                    base_val = avg_data[DIRECTION][metric_type][METHOD][threshold][model][APPROACH_B].get(run)
                    if our_val is not None and base_val is not None:
                        our_vals.append(our_val)
                        base_vals.append(base_val)
            if our_vals and base_vals:
                try:
                    stat, p = wilcoxon(our_vals, base_vals)
                    test_used = "Wilcoxon (Paired)"
                except ValueError:
                    stat, p = mannwhitneyu(our_vals, base_vals, alternative='two-sided')
                    test_used = "Mann-Whitney U (Independent)"
                a12 = compute_a12(our_vals, base_vals)
                stat_rows.append({
                    'Metric': metric_type,
                    'Threshold': threshold, 'Comparison': f"{APPROACH_A} vs {APPROACH_B}",
                    'Test': test_used, 'p-value': sround(p), 'A12': sround(a12), 'N': len(our_vals)
                })


    # --- Save Results to CSV ---
    if not stat_rows:
        print("No statistical results were generated. Check data extraction.")
        return
    stat_df = pd.DataFrame(stat_rows)
    column_order = ['Metric', 'Threshold', 'Comparison', 'Test', 'p-value', 'A12']
    stat_df = stat_df[[col for col in column_order if col in stat_df.columns]]
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    stat_df.to_csv(output_filename, index=False)
    print(f"Saved statistical results to {output_filename}")

# --- Main Execution ---
if avg_data or raw_data_by_id:
    all_methods_present = all(METHOD_CONFIG[metric]["METHOD"] in methods_by_direction.get(METHOD_CONFIG[metric]["DIRECTION"], set()) for metric in METRIC_TYPES)
    if all_methods_present:
        perform_statistical_analysis(f"{APPROACH_A}-vs-{APPROACH_B}.csv")
    else:
        missing_methods = [metric for metric in METRIC_TYPES if METHOD_CONFIG[metric]["METHOD"] not in methods_by_direction.get(METHOD_CONFIG[metric]["DIRECTION"], set())]
        print(f"Methods for {missing_methods} not found in their respective directions.")
else:
    print("No data was extracted. Please check file paths and data structure.")

print("-" * 30)
print("Processing finished.")