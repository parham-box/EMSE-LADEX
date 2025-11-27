import csv
from collections import defaultdict, deque
import numpy as np
from sentence_transformers import SentenceTransformer
import os 

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

class Embedder:
    def __init__(self, model_name="Alibaba-NLP/gte-base-en-v1.5"):
        cache_folder = os.path.join("cache", model_name.replace("/", "_"))
        self.model = SentenceTransformer(model_name, trust_remote_code=True, cache_folder=cache_folder)
        self._cache = {}
    def encode(self, texts):
        to_encode, idxs = [], []
        for i,t in enumerate(texts):
            if t not in self._cache:
                idxs.append(i)
                to_encode.append(t)
        if to_encode:
            embs = self.model.encode(to_encode, normalize_embeddings=True)
            for txt, emb in zip(to_encode, embs):
                self._cache[txt] = emb
        return np.stack([self._cache[t] for t in texts])
    def similarity(self, a,b):
        return self.model.similarity(a, b)

def merge_nodes(graph, node_data):
    rev = defaultdict(list)
    for u, outs in graph.items():
        for v, lab in outs:
            rev[v].append((u, lab))

    in_deg  = {n: len(rev.get(n, [])) for n in node_data}
    out_deg = {n: len(graph.get(n, [])) for n in node_data}

    linear = {n for n in node_data if in_deg[n] <= 1 and out_deg[n] <= 1}

    visited = set()
    chains  = []

    # Extract maximal chains
    for n in node_data:
        if n not in linear or n in visited:
            continue

        # Walk backward to chain start
        cur = n
        while True:
            preds = [p for (p, _) in rev.get(cur, []) if p in linear]
            if len(preds) == 1:
                cur = preds[0]
            else:
                break

        # Walk forward to build the full chain
        chain = []
        while cur in linear and cur not in visited:
            visited.add(cur)
            chain.append(cur)
            succs = [c for (c, _) in graph.get(cur, []) if c in linear]
            if len(succs) == 1:
                cur = succs[0]
            else:
                break

        chains.append(chain)

    # Anything never visited is either non-linear or isolated node
    for n in node_data:
        if n not in visited:
            chains.append([n])

    # Build new nodes and a map old→new
    old2new       = {}
    new_node_data = {}
    for chain in chains:
        if len(chain) == 1:
            nid = chain[0]
            new_node_data[nid] = node_data[nid].copy()
            new_node_data[nid]["original_ids"] = [nid]
        else:
            nid = "_".join(chain)
            name_parts = []
            for i, node in enumerate(chain):
                name_parts.append(node_data[node]["name"])
                if i < len(chain) - 1:
                    next_node = chain[i + 1]
                    for tgt, lab in graph[node]:
                        if tgt == next_node and lab:
                            name_parts.append(f"{lab}")
                            break
            merged_name = " / ".join(name_parts)
            merged_type = node_data[chain[-1]]["type"]
            new_node_data[nid] = {
                "name": merged_name,
                "type": merged_type,
                "original_ids": chain
            }
        for old in chain:
            old2new[old] = nid

    # Rebuild graph skipping old nodes
    new_graph = defaultdict(list)
    for u, outs in graph.items():
        u2 = old2new[u]
        for v, lab in outs:
            v2 = old2new[v]
            if u2 != v2:
                new_graph[u2].append((v2, lab))

    # Deduplicate multiple edges
    for u, outs in new_graph.items():
        seen = set()
        unique = []
        for v, lab in outs:
            key = (v, lab)
            if key not in seen:
                seen.add(key)
                unique.append((v, lab))
        new_graph[u] = unique

    return new_graph, new_node_data

def evaluate_threshold_at_end_one_to_many(csv1, csv2, threshold):
    # 1) Build charts
    g1, nd1, r1, _ = build_chart(csv1)
    g2, nd2, r2, _ = build_chart(csv2)
    g1, nd1 = merge_nodes(g1, nd1)
    g2, nd2 = merge_nodes(g2, nd2)

    # Check structural constraints
    from StructuralConstraintEvaluation import StructuralConstraintEvaluation

    sc1 = StructuralConstraintEvaluation(g1, nd1, r1)
    sc2 = StructuralConstraintEvaluation(g2, nd2, r2)

    violations1 = [k for k,v in sc1.validate().items() if k.startswith("C") and v is False]
    violations2 = [k for k,v in sc2.validate().items() if k.startswith("C") and v is False]

    if violations1 or violations2:
        return {
            "coverage_1_to_2": 0,
            "coverage_2_to_1": 0,
            "accept1": 0,
            "accept2": 0,
            "total1": len(nd1),
            "total2": len(nd2),
            "coverage": 0,
            "aggregated_semantic": 0,
            "final_score": 0,
            "sem_scores": [],
            "unmatched_nodes_chart1": [{"id": n, "name": nd1[n]["name"]} for n in nd1],
            "unmatched_nodes_chart2": [{"id": n, "name": nd2[n]["name"]} for n in nd2],
        }

    # Recompute roots from the pruned node_data / graph
    children1 = {v for outs in g1.values() for (v, _) in outs}
    roots1 = [n for n in nd1 if n not in children1]
    r1 = roots1[0] if roots1 else None
    children2 = {v for outs in g2.values() for (v, _) in outs}
    roots2 = [n for n in nd2 if n not in children2]
    r2 = roots2[0] if roots2 else None

    # 2) Pre-embed all unique labels and node-names
    emb = Embedder()
    all_texts = set(nd1[n]["name"] for n in nd1) | set(nd2[n]["name"] for n in nd2)
    for outs in (*g1.values(), *g2.values()):
        all_texts |= {lab for (_, lab) in outs}
    all_texts = list(all_texts)
    text2emb = dict(zip(all_texts, emb.encode(all_texts)))

    def sim(a, b):
        return float(emb.similarity(text2emb[a], text2emb[b]))

    # Prepare data structures
    all_matches = []
    visited = set()
    queue = deque()

    # Initial root handling with no threshold
    if r1 is not None and r2 is not None:
        root_sim = sim(nd1[r1]["name"], nd2[r2]["name"])
        queue.append((r1, r2, root_sim))

    # BFS with best match selection but no threshold applied here
    while queue:
        u1, u2, score = queue.popleft()
        if (u1, u2) in visited:
            continue
        visited.add((u1, u2))
        all_matches.append((u1, u2, score))

        outs1 = g1.get(u1, [])
        outs2 = g2.get(u2, [])

        for (v1, lab1) in outs1:
            if outs2:
                best_j = None
                best_score = -float("inf")
                for j, (v2, lab2) in enumerate(outs2):
                    s_name = sim(nd1[v1]["name"], nd2[v2]["name"])
                    if lab1 == "" and lab2 == "":
                        combined = s_name
                    else:
                        s_lab = sim(lab1, lab2)
                        combined = (s_lab + s_name) / 2
                    if combined > best_score:
                        best_score = combined
                        best_j = j
                if best_j is not None:
                    v2, _ = outs2[best_j]
                    queue.append((v1, v2, best_score))

    # Apply threshold AFTER traversal
    sem_scores = [score for (_, _, score) in all_matches if score >= threshold]
    matched1 = {u1 for (u1, _, score) in all_matches if score >= threshold}
    matched2 = {u2 for (_, u2, score) in all_matches if score >= threshold}

    # Unmatched
    unmatched1 = [{"id": n, "name": nd1[n]["name"]} for n in nd1 if n not in matched1]
    unmatched2 = [{"id": n, "name": nd2[n]["name"]} for n in nd2 if n not in matched2]

    # Metrics
    accept1 = len(matched1)
    accept2 = len(matched2)
    cov1 = accept1 / len(nd1) if len(nd1) > 0 else 0
    cov2 = accept2 / len(nd2) if len(nd2) > 0 else 0
    coverage = 0.5 * (cov1 + cov2)
    agg_sem = np.mean(sem_scores) if sem_scores else 0
    final_score = (coverage + agg_sem) / 2

    return {
        "coverage_1_to_2": cov1,
        "coverage_2_to_1": cov2,
        "accept1": accept1,
        "accept2": accept2,
        "total1": len(nd1),
        "total2": len(nd2),
        "coverage": coverage,
        "aggregated_semantic": agg_sem,
        "final_score": final_score,
        "sem_scores": str(sem_scores),
        "unmatched_nodes_chart1": unmatched1,
        "unmatched_nodes_chart2": unmatched2
    }

def eval_metrics(gt_path, gen_path):
    """Evaluate a ground truth chart against a generated chart using all evaluation methods."""
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_file = f.read()
    with open(gen_path, "r", encoding="utf-8") as f:
        gen_file = f.read()
    file_id = os.path.basename(gt_path)[:-7]
    print(f"Processing {file_id}: {gt_path} vs {gen_path}")
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    metrics = {}
    try:
        thr_results = {thr: evaluate_threshold_at_end_one_to_many(gt_file, gen_file, thr) for thr in thresholds}
        no_thr = evaluate_threshold_at_end_one_to_many(gt_file, gen_file, 0.0)
    except Exception as e:
        print(f"Error processing {file_id} with evaluate_threshold_at_end_one_to_many: {e}")
        thr_results = {
            thr: {k: (0 if k != 'sem_scores' else []) for k in [
                'coverage_1_to_2', 'coverage_2_to_1', 'aggregated_semantic',
                'accept1', 'accept2', 'total1', 'total2', 'sem_scores'
            ]} for thr in thresholds
        }
        no_thr = thr_results[thresholds[0]]
    for thr in thresholds:
        res = thr_results[thr]
        key = f"evaluate_threshold_at_end_one_to_many_with_threshold_{int(thr*10)}"
        metrics[key] = {
            'coverage_1_to_2': res['coverage_1_to_2'],
            'coverage_2_to_1': res['coverage_2_to_1'],
            'avg_semsim': res['aggregated_semantic'],
            'accept_1': res['accept1'],
            'accept_2': res['accept2'],
            'total_1': res['total1'],
            'total_2': res['total2'],
            'sem_scores': res['sem_scores'],
        }
    metrics[f"evaluate_threshold_at_end_one_to_many_no_threshold"] = {
        'coverage_1_to_2': no_thr['coverage_1_to_2'],
        'coverage_2_to_1': no_thr['coverage_2_to_1'],
        'avg_semsim': no_thr['aggregated_semantic'],
        'accept_1': no_thr['accept1'],
        'accept_2': no_thr['accept2'],
        'total_1': no_thr['total1'],
        'total_2': no_thr['total2'],
        'sem_scores': no_thr['sem_scores'],
    }
    return metrics

def batch_evaluation(gt_dir, gen_dir, output_prefix,di):
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(".txt")]
    eval_method =  "evaluate_threshold_at_end_one_to_many"
    thresholds = ["0.9", "0.8", "0.7", "0.6", "0.5", "no"]
    method_results = {eval_method: []}
    # Process each file pair
    for gt_file in gt_files:
        file_id = gt_file[:-4]
        gt_path = os.path.join(gt_dir, gt_file)
        gen_path = os.path.join(gen_dir, f"{file_id}.txt")
        if not os.path.exists(gen_path):
            print(f"Skipping missing generated file: {gen_path}")
            continue
        metrics = eval_metrics(gt_path, gen_path)
        # Store results for each method
        row = [file_id]
        for thr in thresholds:
            thr_key = int(float(thr)*10) if thr != "no" else "no"
            key = f"{eval_method}_with_threshold_{thr_key}" if thr != "no" else f"{eval_method}_no_threshold"
            m = metrics[key]
            row.extend([
                m['coverage_1_to_2'], m['coverage_2_to_1'], m['avg_semsim'],
                m['accept_1'], m['accept_2'], m['total_1'], m['total_2'],
                m['sem_scores']
            ])
        method_results[eval_method].append(row)

    results_dir = f"{di}/results"
    os.makedirs(results_dir, exist_ok=True)

    # Write separate CSV for each evaluation method
    output_csv = os.path.join(results_dir, f"{os.path.basename(output_prefix)}.csv")
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        header = ["File ID"]
        for thr in thresholds:
            header.extend([
                f"Coverage1 (thr {thr})", f"Coverage2 (thr {thr})",
                f"Avg SemSim (thr {thr})", f"Matched 1 (thr {thr})",
                f"Matched 2 (thr {thr})", f"Total 1 (thr {thr})",
                f"Total 2 (thr {thr})", f"Sem scores (thr {thr})"
            ])
        writer.writerow(header)
        # Write results
        results = method_results[eval_method]
        for row in results:
            writer.writerow(row)
        # Compute and write summary averages
        writer.writerow([])
        for thr in thresholds:
            offset = (thresholds.index(thr) * 8) + 1
            cov1_vals = [row[offset] for row in results]
            cov2_vals = [row[offset+1] for row in results]
            sem_vals = [row[offset+2] for row in results if row[offset+2] != 0]
            avg_cov1 = sum(cov1_vals) / len(cov1_vals) if cov1_vals else 0
            avg_cov2 = sum(cov2_vals) / len(cov2_vals) if cov2_vals else 0
            avg_sem = sum(sem_vals) / len(sem_vals) if sem_vals else 0
            writer.writerow([f"Average thr {thr}", avg_cov1, avg_cov2, avg_sem])
    print(f"Written evaluation for {eval_method} to {output_csv}")

if __name__ == "__main__":
    gt_dir = "../PAGED_Dataset/Acitivty_Diagrams"
    gen_dir = f"../LADEX/Test"
    # GT vs GEN
    out_prefix = f"{gen_dir}_evaluation_results_gt"
    batch_evaluation(gt_dir, gen_dir, out_prefix,gen_dir)
    # GEN vs GT
    out_prefix = f"{gen_dir}_evaluation_results_gen"
    batch_evaluation(gen_dir, gt_dir, out_prefix,gen_dir)
