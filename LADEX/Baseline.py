import os
from dotenv import load_dotenv
import re
import csv
import io
import time
import json
from subprocess import *
from Prompts import (
    get_generate_prompt,
    get_csv_header,
)
import argparse
import importlib
import sys
from pathlib import Path
from openai import OpenAI

load_dotenv(".env")
maxInt = sys.maxsize
csv.field_size_limit(maxInt)
_model = None

def set_model(name: str) -> None:
    global _model
    _model = importlib.import_module(name)
def call_llm_api(messages, model=None, max_tokens=2000, max_completion_tokens=None, temperature=0.0):
    global _model
    api_key = os.getenv("OPENAI_API_KEY")
    client_args = {"api_key": api_key}
    
    client = OpenAI(**client_args)

    model_name = _model

    try:
        request_args = {
            "model": model_name,
            "messages": messages
        }
        if max_completion_tokens is not None: #Reasoning LLM
            request_args["max_tokens"] = max_completion_tokens
        else: #instruction-following LLM
            request_args["max_tokens"] = max_tokens
            request_args["temperature"] = temperature

        response = client.chat.completions.create(**request_args)

        content = response.choices[0].message.content
        usage = getattr(response, "usage", {})

        return content, usage

    except Exception as e:
        print(f"Error calling LLM API: {str(e)}")
        raise

def generate_AD(procedure_text, verbose: bool = False):
    print(f"Generate Activity Diagram:\t Started")
    system_prompt = get_generate_prompt()
    user_prompt = f"Natural-Language description of the procedure:\n{procedure_text}\n\n"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
    ]
    
    start_time = time.time()
    response = call_llm_api(messages)
    end_time = time.time()
    response_text = extract_csv_from_response(response if isinstance(response, str) else response[0])
    
    if verbose:
        print(f"Generate_Activity Diagram - API Response:\n", response_text)
    print(f"Generate Activity Diagram:\t Finished")
    
    token_info = {} if isinstance(response, str) else response[1] if len(response) > 1 else {}
    return {
        "response": response_text,
        "metrics": {
            "duration": end_time - start_time,
            "tokens": token_info
        }
    }

def extract_csv_from_response(response_text):
    lines = response_text.splitlines()
    csv_lines = []
    for line in lines:
        if re.match(r'^\d+,', line.strip()):
            csv_lines.append(line)
        elif csv_lines:
            break
    return "\n".join(csv_lines) if csv_lines else None

def process_procedure(procedure_text, verbose: bool = False):
    metrics = {
        "generate_calls": [],
        "score_calls": [],
        "improve_calls": []
    }
    
    result = generate_AD(procedure_text, verbose=verbose)
    AD = result["response"]
    metrics["generate_calls"].append(result["metrics"])

    return AD, metrics

def process_line(line):
    reader = csv.reader(io.StringIO(line), skipinitialspace=True)
    fields = next(reader)
    # assets_path = find_assets_path(file_path)
    id_field = fields[0].strip().replace(",", " ")
    second_field = fields[1].strip().replace(",", " ")
    third_field = fields[2].strip().replace(",", " ")
    last_field = fields[-1].strip().replace(",", " ")    

    if len(fields) > 4:
        parents_fields = fields[3:-1]
    else:
        parents_fields = []
    
    parents_combined = ",".join(parents_fields)
    tokens = [token for token in re.split(r',', parents_combined) if token.strip()]
    digits_list = []
    for token in tokens:
        nums = re.findall(r'\d+', token)
        if nums:
            digits_list.append("".join(nums))
    parents_processed = ",".join(digits_list)
    
    def quote_if_needed(value):
        # Remove all existing double quotes
        value = value.replace('"', '')
        value = value.replace('\"', '')
        # Wrap with exactly two double quotes
        return f'"{value}"'

    second_field = quote_if_needed(second_field)
    third_field = quote_if_needed(third_field)
    parents_field = quote_if_needed(parents_processed)
    last_field = quote_if_needed(last_field)
    
    res = f"{id_field},{second_field},{third_field},{parents_field},{last_field}"
    return res

def update_chart_ids(AD):
    lines = AD.splitlines()
    records = list(csv.reader(lines))
    ID_IDX = 0
    PARENT_IDX = 3
    max_id = 0
    for row in records:
        id_str = row[ID_IDX]
        if id_str.isdigit():
            max_id = max(max_id, int(id_str))
    id_mapping = {}
    for row in records:
        original_id = row[ID_IDX]
        if not original_id.isdigit():
            max_id += 1
            new_id_str = str(max_id)
            id_mapping[original_id] = new_id_str
            row[ID_IDX] = new_id_str
    for row in records:
        parent_field = row[PARENT_IDX]
        if parent_field:
            for old_id, new_id in id_mapping.items():
                pattern = re.escape(old_id)
                parent_field = re.sub(pattern, new_id, parent_field)
            row[PARENT_IDX] = parent_field
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(records)
    return output.getvalue(), max_id

def save_metrics(metrics, output_file):
    metrics_file = os.path.splitext(output_file)[0] + '_metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_file}")

def process_description(file_path, output_folder, model_output_folder, dataset, verbose=False):
    start_time = time.time()
    metrics = {
        "id": os.path.basename(str(file_path)),
        "total_duration": 0,
        "processed_procedures": []
    }
    
    with open(file_path, "r", encoding="utf-8") as f:
        p = f.read()
    procedure = p
    try:
        data = json.loads(p)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "paragraph" in data[0]:
            procedure = data[0]["paragraph"]
    except json.JSONDecodeError:
        pass

    procedure_metrics = {
        "generate_calls": [],
        "score_calls": [],
        "improve_calls": [],
    }

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(model_output_folder, exist_ok=True)
        
    final_AD, proc_metrics = process_procedure(procedure, verbose)

    procedure_metrics.update(proc_metrics)
    metrics["processed_procedures"].append(procedure_metrics)
    
    final_AD = re.sub(r' +', ' ', final_AD)
    final_AD = final_AD.replace("“", '').replace("”", '')
    final_AD, max_id = update_chart_ids(final_AD)
    id_state = [max_id]
    
    processed_chart = "\n".join(
        process_line(line)
        for line in final_AD.splitlines()
    )
    
    file_path_path = Path(file_path)
    relative_path = file_path_path.relative_to(dataset)
    output_file = os.path.join(model_output_folder, os.path.splitext(relative_path)[0] + '.txt')
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_AD = get_csv_header() + processed_chart
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_AD)
    
    end_time = time.time()
    metrics["total_duration"] = end_time - start_time
    save_metrics(metrics, output_file)
    
    return final_AD

def main():
    parser = argparse.ArgumentParser(description="Run model processing on dataset files.")
    parser.add_argument("--model", default="gpt41mini", help="Model name to use")
    parser.add_argument("--dataset", default="Test", help="Dataset folder")
    parser.add_argument("--output", default="Test", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    set_model(args.model)
    print(f"Using Model:\t{args.model}")

    model_output_folder = os.path.join(args.output, args.model)
    os.makedirs(model_output_folder, exist_ok=True)

    for root, dirs, files in os.walk(args.dataset):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                output_file_name = os.path.splitext(file)[0] + ".txt"
                output_file_path = os.path.join(model_output_folder, output_file_name)

                if os.path.exists(output_file_path):
                    print(f"Skipping {file} (already exists in output).")
                    continue

                final_AD = process_description(
                    file_path=file_path,
                    output_folder=args.output,
                    model_output_folder=model_output_folder,
                    dataset=args.dataset,
                    verbose=args.verbose
                )

if __name__ == "__main__":
    main()
