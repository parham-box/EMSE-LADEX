import os
import csv
import statistics
from collections import defaultdict

# --- Configuration ---
approach = "Industry-LADEX-LLM-NA"
MODELS = ["O4Mini","gpt41mini"]
RUNS = range(1, 6)
THRESHOLDS_ORDER = ["no"]
METRIC_TYPES = ["Completeness", "Correctness"]

# --- Data Storage ---
def nested_dict():
    """Creates a default dictionary that allows for deep nesting."""
    return defaultdict(nested_dict)

all_data = {"Completeness": nested_dict(), "Correctness": nested_dict()}
methods_union = set()

# --- 1. Data Extraction ---
print("Starting data extraction...")
print(f"  Processing Approach: {approach}")
for model_base in MODELS:
    for run in RUNS:
        model_run_prefix = f"{model_base}-{run}"
        base_dir = f"../Results/{approach}/{model_run_prefix}"
        # The results are now directly in a 'results' directory.
        results_dir = os.path.join(base_dir, "results") 

        if not os.path.isdir(results_dir):
            continue
        
        # Directly iterate over files in the results_dir
        for fname in os.listdir(results_dir):
            if not fname.endswith(".csv"):
                continue

            is_gt_vs_gen = "_gt" in fname
            direction = "GT_vs_GEN" if is_gt_vs_gen else "GEN_vs_GT"
            method = fname.replace(f"{model_run_prefix}_evaluation_results_", "").replace(".csv", "")
            methods_union.add(method)

            file_path = os.path.join(results_dir, fname)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Average thr"):
                        parts = line.strip().split(",")
                        threshold = parts[0].replace("Average thr ", "")
                        coverage1 = float(parts[1])
                        
                        if direction == "GT_vs_GEN":
                            all_data["Completeness"][method][threshold][model_base][approach][run] = coverage1
                        else:
                            all_data["Correctness"][method][threshold][model_base][approach][run] = coverage1

print("Data extraction complete.")
print("-" * 30)

# --- 2. Unified Summary Report ---
def create_combined_summary_report(data, methods):
    """Generates a single, integrated CSV summary report."""
    output_filename = f"../Results/{approach}/Summary_Integrated.csv"
    os.makedirs("Results", exist_ok=True)

    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # -- Header --
        header1 = ["Metric", "Threshold", "Model"]
        header1.extend([approach] + [""] * (len(RUNS) - 1))
        header1.extend(["Average", "Std Dev"])
        writer.writerow(header1)

        header2 = ["", "", ""]
        header2.extend([f"run{r}" for r in RUNS])
        header2.extend(["", ""])
        writer.writerow(header2)

        # -- Method Grouping --
        target_methods = [
            "gt",
            "gen"
        ]
        other_methods = sorted([m for m in methods if m not in target_methods])

        all_methods_in_order = [("", target_methods)] + \
                               [(method, [method]) for method in other_methods]

        for section_title, methods_in_section in all_methods_in_order:
            writer.writerow([])
            writer.writerow([f"METHOD: {section_title}"] + [""] * (len(header1) - 1))

            for threshold in THRESHOLDS_ORDER:
                for model in MODELS:
                    for method in methods_in_section:
                        for metric_type in METRIC_TYPES:
                            value_dict = data.get(metric_type, {}).get(method, {}).get(threshold, {}).get(model, {})

                            if not value_dict:  # Skip if no data for this combination
                                continue
                            
                            data_cells = []
                            numerical_values = []
                            for run in RUNS:
                                value = value_dict.get(approach, {}).get(run, None)
                                if value is not None:
                                    formatted_value = round(float(value) * 100, 2)
                                    numerical_values.append(formatted_value)
                                    data_cells.append(f"{formatted_value:.2f}")
                                else:
                                    data_cells.append("N/A")

                            # Don't write a row if it contains no actual data
                            if not numerical_values:
                                continue

                            # Calculate average and standard deviation
                            avg = round(statistics.mean(numerical_values), 2) if numerical_values else "N/A"
                            std_dev = round(statistics.stdev(numerical_values), 2) if len(numerical_values) > 1 else "N/A"
                            
                            row_prefix = f"{model}-{metric_type}" if section_title.startswith("Combined") else metric_type
                            row = [row_prefix, threshold, model] + data_cells + [avg, std_dev]
                            writer.writerow(row)

    print(f"Successfully created summary file: {output_filename}")


# --- 3. Create Summary File ---
if any(all_data[metric] for metric in METRIC_TYPES):
    create_combined_summary_report(all_data, methods_union)
else:
    print("No data was extracted. Please check your file paths and structure.")
    print("Expected path format: Results/{approach}/{model}-{run}/results/{filename}.csv")

print("-" * 30)
print("Processing finished.")