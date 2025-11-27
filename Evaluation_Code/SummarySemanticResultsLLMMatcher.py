import os
import csv
import statistics
from collections import defaultdict

APPROACHES = ["Public-LADEX-LLM-NA"]
MODELS = ["O4Mini", "gpt41mini","local"]
RUNS = range(1, 6)
METRIC_TYPES = ["Completeness", "Correctness"]

# Map new filenames to their metric and the "method" name expected by the report
FILE_METHOD_MAP = {
    "llm_judge_results_gt_to_gen.csv": {
        "metric": "Completeness",
        "method": "LLM Matcher"
    },
    "llm_judge_results_gen_to_gt.csv": {
        "metric": "Correctness",
        "method": "LLM Matcher"
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
            results_dir = f"../Results/{approach}/{model_run_prefix}/llm-as-judge-results"

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

print("Data extraction complete.")
print(f"Methods found: {methods_union}")
print("-" * 30)


# --- 2. Summary Report ---
def create_combined_summary_report(all_data, methods_union):
    base_dir = f"../Results/{APPROACHES[0]}"
    os.makedirs(base_dir, exist_ok=True)
    output_filename = os.path.join(base_dir, "Summary_Integrated_llm_matcher.csv")

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


# --- 3. Create Summary File ---
if any(all_data[metric] for metric in METRIC_TYPES):
    create_combined_summary_report(all_data, methods_union)
else:
    print("No data was extracted. Please check your file paths and structure.")
    print("Expected path format: ../Results/{approach}/{model}-{run}/llm-as-judge-../Results/...csv")

print("-" * 30)
print("Processing finished.")
