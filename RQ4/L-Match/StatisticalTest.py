import os
import csv
import statistics
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import os
import numpy as np
import pandas as pd

#load all data
APPROACHES = [
    "Industry-Baseline",
    "Industry-LADEX-ALG-LLM",
    "Industry-LADEX-ALG-NA",
    "Industry-LADEX-LLM-LLM",
    "Industry-LADEX-LLM-NA",
    "Public-Baseline",
    "Public-LADEX-ALG-LLM",
    "Public-LADEX-ALG-NA",
    "Public-LADEX-LLM-LLM",
    "Public-LADEX-LLM-NA"
    ]
MODELS = ["O4Mini", "gpt41mini","local"]
RUNS = range(1, 6)
METRIC_TYPES = ["Completeness", "Correctness"]
#chose what two variants you want to compare
RQ = "RQ4"
m1 = "Public-LADEX-LLM-NA"
m2 = "Public-LADEX-ALG-LLM"
FILE_METHOD_MAP = {
    "llm_judge_results_gt_to_gen.csv": {
        "metric": "Completeness",
        "method": "LLM_Matcher"
    },
    "llm_judge_results_gen_to_gt.csv": {
        "metric": "Correctness",
        "method": "LLM_Matcher"
    }
}

# --- Data Storage ---
def nested_dict():
    return defaultdict(nested_dict)

all_data = {"Completeness": nested_dict(), "Correctness": nested_dict()}
methods_union = set()

# --- 1. Data Extraction ---
print("Starting data extraction...")
for approach in APPROACHES:
    print(f"  Processing Approach: {approach}")
    for model_base in MODELS:
        for run in RUNS:
            model_run_prefix = f"{model_base}-{run}"
            results_dir = f"../../Results/{approach}/{model_run_prefix}/llm-as-judge-results"

            if not os.path.isdir(results_dir):
                print(f"    Skipping (not found): {results_dir}")
                continue

            # Loop through the specific files we care about
            for fname, info in FILE_METHOD_MAP.items():
                file_path = os.path.join(results_dir, fname)
                if not os.path.isfile(file_path):
                    print(f"      File not found: {file_path}")
                    continue

                metric_type = info["metric"]
                method = info["method"]
                methods_union.add(method)

                found_average = False
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if not row:
                                continue
                            # Find the AVERAGE line
                            if row[0] == "AVERAGE" and len(row) > 3 and row[3].startswith("Average of coverage_1_to_2:"):
                                try:
                                    value_str = row[3].split(":")[1].strip()
                                    coverage_value = float(value_str)
                                    all_data[metric_type][method][model_base][approach][run] = coverage_value
                                    found_average = True
                                    break
                                except (ValueError, IndexError, TypeError) as e:
                                    print(f"      Error parsing average value in: {file_path}. Error: {e}")
                except Exception as e:
                    print(f"    Error reading file {file_path}: {e}")

                if not found_average:
                    print(f"      'AVERAGE' line not found or parsed correctly in: {file_path}")


def compute_a12(our, base):
    if not our or not base: return 0
    count = ties = 0
    for o in our:
        for b in base:
            if o > b: count += 1
            elif o == b: ties += 1
    total = len(our) * len(base)
    return (count + 0.5 * ties) / total if total > 0 else 0

# --- 2. Summary Report ---
def create_combined_summary_report(all_data, methods_union):
    base_dir = f"../../Results/{APPROACHES[0]}"
    os.makedirs(base_dir, exist_ok=True)
    output_filename = os.path.join(base_dir, "Summary_Integrated_llm_as_judge.csv")

    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header1 = ["Metric", "Model"]
        for approach in APPROACHES:
            header1.extend([f"{approach} run{r}" for r in RUNS])
        header1.extend(["Average", "Std Dev"])
        writer.writerow(header1)

        # Write all available data
        for method in sorted(methods_union):
            writer.writerow([])
            writer.writerow([f"METHOD: {method}"] + [""] * (len(header1) - 1))

            for model in MODELS:
                for metric_type in METRIC_TYPES:
                    metric_data = all_data.get(metric_type, {}).get(method, {}).get(model, {})
                    if not metric_data:
                        continue

                    data_cells = []
                    numerical_values = []
                    for approach in APPROACHES:
                        for run in RUNS:
                            value = metric_data.get(approach, {}).get(run, None)
                            if value is not None:
                                formatted_value = round(value * 100, 2)
                                data_cells.append(f"{formatted_value:.2f}")
                                numerical_values.append(formatted_value)
                            else:
                                data_cells.append("N/A")

                    if not numerical_values:
                        continue

                    avg = round(statistics.mean(numerical_values), 2)
                    std_dev = round(statistics.stdev(numerical_values), 2) if len(numerical_values) > 1 else "N/A"

                    row = [f"{model}-{metric_type}", model] + data_cells + [avg, std_dev]
                    writer.writerow(row)

    print(f"Successfully created summary file: {output_filename}")

def stat_test(all_data, methods_union):
    for method in sorted(methods_union):
        for metric_type in METRIC_TYPES:
            stats_records = []
            stats_path = f"../../{RQ}/{method}_{metric_type}_Stats_{m1}_vs_{m2}.csv"

            vals_base_all = []
            vals_our_all = []

            for model2 in MODELS:
                vals_base_all.extend([
                    all_data[metric_type][method][model2][m2][r] * 100
                    for r in RUNS
                    if r in all_data[metric_type][method][model2][m2]
                ])
                vals_our_all.extend([
                    all_data[metric_type][method][model2][m1][r] * 100
                    for r in RUNS
                    if r in all_data[metric_type][method][model2][m1]
                ])

            if vals_base_all and vals_our_all:
                try:
                    if len(vals_base_all) == len(vals_our_all):
                        stat, p = wilcoxon(vals_our_all, vals_base_all)
                        test_used = "Wilcoxon (paired)"
                except ValueError:
                    stat, p, test_used = np.nan, np.nan, "Error"
                a12 = compute_a12(vals_our_all, vals_base_all)
            else:
                stat, p, a12, test_used = np.nan, np.nan, np.nan, "N/A"
            stats_records.append({
                "Method": method,
                "Metric": metric_type,
                "Models Combined": ", ".join(MODELS),
                "Test": test_used,
                "p_value": p,
                "A12": a12,
            })


            df_stats = pd.DataFrame(stats_records)
            df_stats.to_csv(stats_path, index=False)
            print(f"Saved stats: {stats_path}")

if any(all_data[metric] for metric in METRIC_TYPES):
    stat_test(all_data, methods_union)
else:
    print("No data available for plotting.")
