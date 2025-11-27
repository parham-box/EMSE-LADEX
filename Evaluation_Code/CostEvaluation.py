import os
import json
import re
import csv

avf = ["gpt41mini-", "O4Mini-"]
approach = 'Industry-LADEX-LLM-NA'

def extract_token_values(token_str):
    try:
        completion_tokens = int(re.search(r"completion_tokens=(\d+)", token_str).group(1))
        prompt_tokens = int(re.search(r"prompt_tokens=(\d+)", token_str).group(1))
        total_tokens = int(re.search(r"total_tokens=(\d+)", token_str).group(1))
        reasoning_tokens = int(re.search(r"reasoning_tokens=(\d+)", token_str).group(1))
        output_tokens = completion_tokens - reasoning_tokens
        return output_tokens, reasoning_tokens, completion_tokens, prompt_tokens, total_tokens
    except Exception as e:
        print(f"Error extracting tokens: {e}")
        return 0, 0, 0, 0, 0

results = []

for cat in avf:
    total_files = 0
    total_calls = 0

    # Token totals across all files and calls
    output_tokens_sum = 0
    reasoning_tokens_sum = 0
    completion_tokens_sum = 0
    prompt_tokens_sum = 0
    total_tokens_sum = 0

    for i in range(1, 6):
        model_dir = f"../Results/{approach}/{cat}{i}"
        if not os.path.exists(model_dir):
            continue

        for file_name in os.listdir(model_dir):
            if file_name.endswith("_metrics.json"):
                file_path = os.path.join(model_dir, file_name)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        proc = data.get("processed_procedures", [{}])[0]

                        file_has_calls = False

                        for call_type in ["generate_calls", "score_calls", "improve_calls"]:
                            calls = proc.get(call_type, [])
                            if calls:
                                file_has_calls = True
                            total_calls += len(calls)
                            for call in calls:
                                token_str = call.get("tokens", "")
                                o, r, c, p, t = extract_token_values(token_str)
                                output_tokens_sum += o
                                reasoning_tokens_sum += r
                                completion_tokens_sum += c
                                prompt_tokens_sum += p
                                total_tokens_sum += t

                        if file_has_calls:
                            total_files += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if total_files > 0 and total_calls > 0:
        avg_calls_per_file = total_calls / total_files
        results.append({
            "Model": cat,
            "Total_Files": total_files,
            "Avg_Calls_per_File": f"{avg_calls_per_file:.2f}",
            "Avg_Prompt_Tokens_per_Call": f"{prompt_tokens_sum / total_calls:.2f}",
            "Avg_Output_Tokens_per_Call": f"{output_tokens_sum / total_calls:.2f}",
            "Avg_Reasoning_Tokens_per_Call": f"{reasoning_tokens_sum / total_calls:.2f}",
            # "Avg_Completion_Tokens_per_Call": f"{completion_tokens_sum / total_calls:.2f}",
            # "Avg_Total_Tokens_per_Call": f"{total_tokens_sum / total_calls:.2f}",
        })
    else:
        results.append({
            "Model": cat,
            "Total_Files": 0,
            "Avg_Calls_per_File": 0,
            "Avg_Prompt_Tokens_per_Call": 0,
            "Avg_Output_Tokens_per_Call": 0,
            "Avg_Reasoning_Tokens_per_Call": 0,
        })

# Write results to CSV
output_dir = f"../Results/{approach}"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cost_evaluation.csv")

with open(output_path, "w", newline="") as csvfile:
    fieldnames = [
        "Model",
        "Total_Files",
        "Avg_Calls_per_File",
        "Avg_Prompt_Tokens_per_Call",
        "Avg_Output_Tokens_per_Call",
        "Avg_Reasoning_Tokens_per_Call"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n Results saved to {output_path}")
