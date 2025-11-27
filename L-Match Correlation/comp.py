import os
import json
import pandas as pd

def normalize_pair(pair):
    a = str(pair[0]).strip()
    b = str(pair[1]).strip()
    return tuple(sorted([a, b]))


def compute_metrics(human_pairs, llm_pairs):
    gt_set = {normalize_pair(p) for p in human_pairs}
    pred_set = {normalize_pair(p) for p in llm_pairs}

    tp_set = gt_set & pred_set
    fp_set = pred_set - gt_set
    fn_set = gt_set - pred_set

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def process_dataset(keyword):
    results_list = []

    human_dir = f"./Human-Matchings/{keyword}"
    llm_dir = f"./L-Match/{keyword}"

    files = sorted(
        [f for f in os.listdir(human_dir) if f.endswith(".json")],
        key=lambda x: int(x.split(".")[0])
    )

    print(f"\nProcessing dataset: {keyword}")
    print(f"Found {len(files)} items.")

    for fname in files:
        base = fname.replace(".json", "")

        # Load human
        with open(os.path.join(human_dir, fname), "r") as f:
            human_data = json.load(f)

        # LLM predictions
        llm_gt_path = os.path.join(llm_dir, f"{base}_gt_to_gen.json")
        llm_gen_path = os.path.join(llm_dir, f"{base}_gen_to_gt.json")

        if not (os.path.exists(llm_gt_path) and os.path.exists(llm_gen_path)):
            print(f"Skipping {base}, missing LLM prediction files.")
            continue

        with open(llm_gt_path, "r") as f:
            llm_gt_to_gen = json.load(f)

        with open(llm_gen_path, "r") as f:
            llm_gen_to_gt = json.load(f)

        # Compute both directions
        res_gt = compute_metrics(human_data["gt_to_gen"], llm_gt_to_gen)
        res_gen = compute_metrics(human_data["gen_to_gt"], llm_gen_to_gt)

        results_list.append(res_gt)
        results_list.append(res_gen)

    return results_list

def aggregate(results_list):
    if not results_list:
        return {"Precision": 0, "Recall": 0, "F1": 0}

    precision_vals = [r["Precision"] for r in results_list]
    recall_vals = [r["Recall"] for r in results_list]
    f1_vals = [r["F1"] for r in results_list]

    return {
        "Precision": sum(precision_vals) / len(precision_vals),
        "Recall": sum(recall_vals) / len(recall_vals),
        "F1": sum(f1_vals) / len(f1_vals)
    }


print("\n--- Starting Matching Evaluation Script ---")

# Industry
industry_results = process_dataset("Industry")
industry_scores = aggregate(industry_results)

# Paged
paged_results = process_dataset("Paged")
paged_scores = aggregate(paged_results)

final_df = pd.DataFrame([
    {"Dataset": "Industry", **industry_scores},
    {"Dataset": "Paged", **paged_scores}
])

final_df.to_csv("./final_scores.csv", index=False)

print("\nSaved final_scores.csv")
print("\n--- Done ---")
