from dotenv import load_dotenv
import csv
import time
import json
from subprocess import *
import importlib
from pathlib import Path
from collections import defaultdict
from csv import DictWriter
from StructuralConstraintEvaluation import StructuralConstraintEvaluation
from openai import OpenAI
import os

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
def build_chart(csv_text):
    reader = csv.DictReader(line for line in csv_text.strip().splitlines() if not line.startswith("#"))
    graph, node_data, children = defaultdict(list), {}, set()
    for row in reader:
        nid = row["id"].strip()
        node_data[nid] = {
            "name": row["name"].strip(),
            "type": row["type"].strip()
        }
        for p in row["parent"].split(","):
            p = p.strip()
            if not p:
                continue
            graph[p].append((nid, row["transition_label"].strip()))
            children.add(nid)
    roots = list(set(node_data) - children)
    root = roots[0] if roots else None
    total_edges = sum(len(v) for v in graph.values())
    return graph, node_data, root, total_edges

def _get_default_metrics(total1=0, total2=0, unmatched1=None, unmatched2=None):
    return {
        "coverage_1_to_2": 0,
        "coverage_2_to_1": 0,
        "accept1": 0,
        "accept2": 0,
        "total1": total1,
        "total2": total2,
        "unmatched_nodes_chart1": unmatched1 if unmatched1 is not None else [],
        "unmatched_nodes_chart2": unmatched2 if unmatched2 is not None else [],
    }

def evaluate(diagram_gt_path: str = "", diagram_gen_path: str = "", api_key: str = None, max_tokens: int = 4096, verbose: bool = False) -> dict:
    print(f"LLM Matcher Evaluation: Comparing '{diagram_gt_path}' vs '{diagram_gen_path}'")
    try:
        diagram_gt = Path(diagram_gt_path).read_text(encoding="utf-8")
        diagram_gen = Path(diagram_gen_path).read_text(encoding="utf-8")
    except FileNotFoundError as e:
        print(f"LLM Matcher Evaluation: Error reading files: {e}")
        return {
            "response": f"Error reading file: {e}",
            "error": True,
            "metrics_gt_to_gen": _get_default_metrics(),
            "metrics_gen_to_gt": _get_default_metrics(),
            "common_metrics": {
            }
        }
    except Exception as e:
        print(f"LLM Matcher Evaluation: Error reading files: {e}")
        return {
            "response": f"Error reading file: {e}",
            "error": True,
            "metrics_gt_to_gen": _get_default_metrics(),
            "metrics_gen_to_gt": _get_default_metrics(),
            "common_metrics": {}
        }

    try:
        g1, nd1, r1, _ = build_chart(diagram_gt)
        g2, nd2, r2, _ = build_chart(diagram_gen)
        
        total1 = len(nd1) # This is total_GT
        total2 = len(nd2) # This is total_GEN
        
        violations1 = [k for k,v in StructuralConstraintEvaluation(g1, nd1, r1).validate().items() if k.startswith("C") and v is False]
        violations2 = [k for k,v in StructuralConstraintEvaluation(g2, nd2, r2).validate().items() if k.startswith("C") and v is False]

    except Exception as e:
        print(f"LLM Matcher Evaluation: Error during graph build/check: {e}")
        return {
            "response": f"Error during graph build/check: {e}",
            "error": True,
            "metrics_gt_to_gen": _get_default_metrics(),
            "metrics_gen_to_gt": _get_default_metrics(),
            "common_metrics": {
            }
        }

    if violations1 or violations2:
        print("LLM Matcher Evaluation:\t Structural violation found. Skipping LLM call.")
        unmatched1 = [{"id": n, "name": nd1[n]["name"]} for n in nd1]
        unmatched2 = [{"id": n, "name": nd2[n]["name"]} for n in nd2]
        
        default_metrics = _get_default_metrics(total1, total2, unmatched1, unmatched2)
        
        return {
            "response": "",
            "error": True,
            "metrics_gt_to_gen": default_metrics,
            "metrics_gen_to_gt": default_metrics,
            "common_metrics": {
                "total1": total1,
                "total2": total2,
            }
        }

    print("LLM Matcher Evaluation:\t No structural violations. Calling LLM.")
    system_prompt = """You are an AI assistant specializing in diagram analysis. Your task is to semantically and behaviouraly match nodes between two diagrams provided by the user.

The user will provide two diagrams in CSV format: 'gt' (ground truth) and 'gen' (generated). The CSVs include columns like 'id', 'name', 'type', 'parent' , and'transition_label'.

Your task is to:
1.  Read both CSVs.
2.  Perform a matching between the nodes from 'gt' to 'gen' and from 'gen' to 'gt'. The match should be based on the meaning and value of the 'name' and 'transition_label' columns, the behavior of the diagram, the similarity of predecessors and successors, and finally, the 'type' column, and not just the 'id' column. Similar labeling and behavioral similarity are more important than the 'type' column. 
3.  A single node in one diagram may match multiple nodes in the other.
4.  You must return **only** a single JSON object containing the results. Do not include any explanatory text, apologies, or conversational wrappers.

The output must be a single JSON object with the exact following structure:
{
"gt_to_gen": [
    ["gt_id_str_1", "gen_id_str_a"],
    ["gt_id_str_2", "gen_id_str_b"],
    ...
],
"gen_to_gt": [
    ["gen_id_str_a", "gt_id_str_1"],
    ["gen_id_str_b", "gt_id_str_2"],
    ...
]
}
One-Shot Example:
gt:
id,name,type,parent,transition_label
1,"start","start","",
2,"check and approve request","entity","1",
3,"end","end","2",

gen:
id,name,type,parent,transition_label
a,"begin","start","",
b,"check the request","entity","a",
c,"approve the request","entity","b",
d,"finish","end","c",

Example Output JSON
{
  "gt_to_gen": [
    ["1", "a"],
    ["2", "b"],
    ["2", "c"],
    ["3", "d"]
  ],
  "gen_to_gt": [
    ["a", "1"],
    ["b", "2"],
    ["c", "2"],
    ["d", "3"]
  ]
}
"""
    user_prompt = f"Diagram gt:\n{diagram_gt} \n\n----\n\nDiagram gen:\n{diagram_gen}"
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    
    response = call_llm_api(messages, max_tokens=max_tokens, temperature=0.0)
    
    if isinstance(response, tuple):
        response_text, token_info = response
    else:
        response_text = response
    
    common_metrics = {
        "total1": total1,
        "total2": total2,
    }

    try:
        response_json = json.loads(response_text)
        gt_to_gen_matches = response_json.get("gt_to_gen", [])
        gen_to_gt_matches = response_json.get("gen_to_gt", [])
        
        matched1_g2g = {gt_id for (gt_id, gen_id) in gt_to_gen_matches}
        matched2_g2g = {gen_id for (gt_id, gen_id) in gt_to_gen_matches}
        unmatched1_g2g = [{"id": n, "name": nd1[n]["name"]} for n in nd1 if n not in matched1_g2g]
        unmatched2_g2g = [{"id": n, "name": nd2[n]["name"]} for n in nd2 if n not in matched2_g2g]
        accept1_g2g = len(matched1_g2g) # Matched GT
        accept2_g2g = len(matched2_g2g) # Matched GEN
        cov1_g2g = accept1_g2g / total1 if total1 > 0 else 0 # Matched GT / Total GT
        cov2_g2g = accept2_g2g / total2 if total2 > 0 else 0 # Matched GEN / Total GEN

        metrics_gt_to_gen = {
            "coverage_1_to_2": cov1_g2g,
            "coverage_2_to_1": cov2_g2g,
            "accept1": accept1_g2g,
            "accept2": accept2_g2g,
            "unmatched_nodes_chart1": unmatched1_g2g,
            "unmatched_nodes_chart2": unmatched2_g2g,
        }


        matched_gen_nodes = {gen_id for (gen_id, gt_id) in gen_to_gt_matches} 
        matched_gt_nodes = {gt_id for (gen_id, gt_id) in gen_to_gt_matches}

        unmatched_gen_nodes = [{"id": n, "name": nd2[n]["name"]} for n in nd2 if n not in matched_gen_nodes] 
        unmatched_gt_nodes = [{"id": n, "name": nd1[n]["name"]} for n in nd1 if n not in matched_gt_nodes]

        accept_gen = len(matched_gen_nodes)
        accept_gt = len(matched_gt_nodes)

        cov_gen = accept_gen / total2 if total2 > 0 else 0
        cov_gt = accept_gt / total1 if total1 > 0 else 0
        
        metrics_gen_to_gt = {
            "coverage_1_to_2": cov_gen,  
            "coverage_2_to_1": cov_gt,  
            "accept1": accept_gen,     
            "accept2": accept_gt,    
            "unmatched_nodes_chart1": unmatched_gen_nodes, 
            "unmatched_nodes_chart2": unmatched_gt_nodes, 
        }


        print("LLM Matcher Evaluation:\t Finished")
        
        return {
            "response": response_text,
            "error": False,
            "metrics_gt_to_gen": metrics_gt_to_gen,
            "metrics_gen_to_gt": metrics_gen_to_gt,
            "common_metrics": common_metrics
        }

    except json.JSONDecodeError as e:
        print(f"LLM Matcher Evaluation: Failed to decode LLM JSON response: {e}")
        unmatched1 = [{"id": n, "name": nd1[n]["name"]} for n in nd1]
        unmatched2 = [{"id": n, "name": nd2[n]["name"]} for n in nd2]
        default_metrics = _get_default_metrics(total1, total2, unmatched1, unmatched2)
        
        return {
            "response": response_text, # Return faulty text for debugging
            "error": True,
            "metrics_gt_to_gen": default_metrics,
            "metrics_gen_to_gt": default_metrics,
            "common_metrics": {
                **common_metrics,
            }
        }
    except Exception as e:
        print(f"LLM Matcher Evaluation: Error during metric calculation: {e}")
        return {
            "response": response_text,
            "error": True,
            "metrics_gt_to_gen": _get_default_metrics(total1, total2),
            "metrics_gen_to_gt": _get_default_metrics(total1, total2),
            "common_metrics": {
                **common_metrics,
            }
        }
def _save_metrics_to_csv(results_data: list, output_csv_path: Path):
    if not results_data:
        print(f"No results to save to {output_csv_path}")
        return

    flat_results = []
    for metrics in results_data:
        flat_m = metrics.copy()
                    
        flat_m["unmatched_nodes_chart1"] = len(flat_m.get("unmatched_nodes_chart1", []))
        flat_m["unmatched_nodes_chart2"] = len(flat_m.get("unmatched_nodes_chart2", []))
        
        flat_m.pop("sem_scores", None)
        flat_m.pop("response", None)
        
        flat_results.append(flat_m)

    all_fieldnames = set()
    for row in flat_results:
        all_fieldnames.update(row.keys())
    
    ordered_fieldnames = sorted(list(all_fieldnames))
    if "file" in ordered_fieldnames:
        ordered_fieldnames.remove("file")
        ordered_fieldnames.insert(0, "file")

    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = DictWriter(csvfile, fieldnames=ordered_fieldnames)
            writer.writeheader()
            for row in flat_results:
                full_row = {field: row.get(field, "") for field in ordered_fieldnames}
                writer.writerow(full_row)
            
            if flat_results:
                writer.writerow({})
                
                summary_row = {field: "" for field in ordered_fieldnames}
                summary_row[ordered_fieldnames[0]] = "AVERAGE"
                
                cols_to_average = [
                    'coverage_1_to_2', 'coverage_2_to_1',
                ]
                c12av = 0
                c21av = 0
                for col in cols_to_average:
                    if col not in ordered_fieldnames:
                        continue 
                    
                    values = []
                    for row in flat_results:
                        val = row.get(col)
                        try:
                            if val is not None:
                                values.append(float(val))
                        except (ValueError, TypeError):
                            continue 
                            
                    if values:
                        avg = sum(values) / len(values)
                        summary_row[col] = f"Average of {col}: {avg:.4f}"
                    else:
                        summary_row[col] = "N/A"
                writer.writerow(summary_row)
            
        print(f"LLM Judge results saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV to {output_csv_path}: {e}")
        
def evaluate_directory_comparison(gt_dir: str, gen_dir: str, **kwargs):
    print(f"\n--- Starting Evaluation for: {gen_dir} ---")
    results_data_gt_to_gen = []
    results_data_gen_to_gt = []
    
    gt_path = Path(gt_dir)
    gen_path = Path(gen_dir)

    results_output_dir = gen_path / "llm-as-judge-results"
    results_output_dir.mkdir(parents=True, exist_ok=True)
    
    gt_files = list(gt_path.glob("*.txt"))
    if not gt_files:
        print(f"No .txt files found in ground truth directory: {gt_path}")
        return

    for gt_file_path in gt_files:
        file_name = gt_file_path.name
        gen_file_path = gen_path / file_name
        
        base = file_name.replace(".txt", "")


        if not gen_file_path.exists():
            print(f"Warning: Skipping {file_name}, corresponding file not found in {gen_dir}")
            continue
        
        eval_result = evaluate(
            diagram_gt_path=str(gt_file_path),
            diagram_gen_path=str(gen_file_path),
            api_key=kwargs.get("api_key"),
            max_tokens=kwargs.get("max_tokens", 4096),
            verbose=kwargs.get("verbose", False),
        )        
        common_metrics = eval_result.get("common_metrics", {})
        metrics_g2g = eval_result.get("metrics_gt_to_gen", {})
        metrics_g2t = eval_result.get("metrics_gen_to_gt", {})
        row_g2g = {**common_metrics, **metrics_g2g, "file": file_name}
        
        common_metrics_g2t = common_metrics.copy()
        common_metrics_g2t["total1"] = common_metrics.get("total2") 
        common_metrics_g2t["total2"] = common_metrics.get("total1") 
        row_g2t = {**common_metrics_g2t, **metrics_g2t, "file": file_name}


        if not eval_result.get("error"):
            try:
                response_obj = json.loads(eval_result["response"])

                out_gt_to_gen = results_output_dir / f"{base}_gt_to_gen.json"
                out_gen_to_gt = results_output_dir / f"{base}_gen_to_gt.json"
                out_gt_to_gen.write_text(
                    json.dumps(response_obj.get("gt_to_gen", []), indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                out_gen_to_gt.write_text(
                    json.dumps(response_obj.get("gen_to_gt", []), indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )

                print(f"Saved match files: {out_gt_to_gen.name}, {out_gen_to_gt.name}")

            except Exception as e:
                print(f"Error saving split JSON results for {file_name}: {e}")

        results_data_gt_to_gen.append(row_g2g)
        results_data_gen_to_gt.append(row_g2t)

    _save_metrics_to_csv(
        results_data_gt_to_gen,
        results_output_dir / "llm_judge_results_gt_to_gen.csv"
    )
    
    _save_metrics_to_csv(
        results_data_gen_to_gt,
        results_output_dir / "llm_judge_results_gen_to_gt.csv"
    )

if __name__ == "__main__":
    load_dotenv() 
    
    set_model('O4Mini')

    variants = ["Public-LADEX-ALG-LLM"]
    llms = ["gpt41mini-","O4Mini-","local-"]
    GT_DIR = "../Dataset/PAGED_Dataset/Acitivty_Diagrams" 

    for v in variants:
        for l in llms:
            for i in range(1, 6):
                gen_dir = f"../Results/{v}/{l}{i}"
                
                if not Path(gen_dir).is_dir():
                    print(f"Skipping non-existent directory: {gen_dir}")
                    continue
                    
                evaluate_directory_comparison(
                    gt_dir=GT_DIR,
                    gen_dir=gen_dir,
                )
