import os
import csv
from collections import defaultdict, deque

class StructuralConstraintEvaluation:
    def __init__(self, graph, node_data, root):
        self.graph = graph
        self.node_data = node_data
        self.root = root
        self.rev_graph = self._build_reverse_graph()

    def _build_reverse_graph(self):
        rev = {n: [] for n in self.node_data}
        for u, outs in self.graph.items():
            for v, _ in outs:
                rev[v].append(u)
        return rev

    def validate(self):
        results = {}
        # C1: Exactly one start node
        start_nodes = [n for n, d in self.node_data.items() if d["type"].lower() == "start"]
        results["C1"] = len(start_nodes) == 1
        results["C1_nodes"] = start_nodes

        # C2: At least one end node
        end_nodes = [n for n, d in self.node_data.items() if d["type"].lower() == "end"]
        results["C2"] = len(end_nodes) >= 1
        results["C2_nodes"] = end_nodes

        # C3: Start node has no incoming transitions
        c3_invalid = [n for n in start_nodes if self.rev_graph[n]]
        results["C3"] = len(c3_invalid) == 0
        results["C3_nodes"] = c3_invalid

        # C4: End nodes have no outgoing transitions
        c4_invalid = [n for n in end_nodes if self.graph.get(n)]
        results["C4"] = len(c4_invalid) == 0
        results["C4_nodes"] = c4_invalid

        # C5: Decision nodes have at least two outgoing transitions
        decision_nodes = [n for n, d in self.node_data.items() if d["type"].lower() == "condition"]
        c5_invalid = [n for n in decision_nodes if len(self.graph.get(n, [])) < 2]
        results["C5"] = len(c5_invalid) == 0
        results["C5_nodes"] = c5_invalid

        # C6: Fully connected from start node
        reachable = set()
        if len(start_nodes) == 1:
            from collections import deque
            queue = deque([start_nodes[0]])
            while queue:
                node = queue.popleft()
                if node in reachable:
                    continue
                reachable.add(node)
                for neighbor, _ in self.graph.get(node, []):
                    queue.append(neighbor)
        disconnected_nodes = set(self.node_data.keys()) - reachable
        results["C6"] = len(disconnected_nodes) == 0
        results["C6_nodes"] = list(disconnected_nodes)

        return results

def build_chart(csv_text):
    reader = csv.DictReader(
        line for line in csv_text.strip().splitlines() if not line.startswith("#")
    )
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


def validate_directory(diagram_dir):
    validation_results = []
    for filename in os.listdir(diagram_dir):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(diagram_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            csv_text = f.read()

        try:
            graph, node_data, root, _ = build_chart(csv_text)
            validator = StructuralConstraintEvaluation(graph, node_data, root)
            results = validator.validate()
            print(f"{filename}:")
            for key in ["C1", "C2", "C3", "C4", "C5", "C6"]:
                print(f"  {key}: {results[key]}")
                if not results[key]:
                    print(f"    Violating nodes: {results[key + '_nodes']}")
            print()

            # Collect results for CSV output
            validation_results.append({
                "file": filename,
                "C1": results["C1"],
                "C1_violating_nodes": results["C1_nodes"] if not results["C1"] else '',
                "C2": results["C2"],
                "C2_violating_nodes": results["C2_nodes"] if not results["C2"] else '',
                "C3": results["C3"],
                "C3_violating_nodes": results["C3_nodes"] if not results["C3"] else '',
                "C4": results["C4"],
                "C4_violating_nodes": results["C4_nodes"] if not results["C4"] else '',
                "C5": results["C5"],
                "C5_violating_nodes": results["C5_nodes"] if not results["C5"] else '',
                "C6": results["C6"],
                "C6_violating_nodes": results["C6_nodes"] if not results["C6"] else '',
            })
        except Exception as e:
            validation_results.append({"file": filename, "error": str(e)})

    # Output results to CSV
    output_csv_path = os.path.join(diagram_dir, "results/structural_constraints.csv")
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow([
            "file", "C1", "C1_violating_nodes", "C2", "C2_violating_nodes", "C3",
            "C3_violating_nodes", "C4", "C4_violating_nodes", "C5", "C5_violating_nodes",
            "C6", "C6_violating_nodes", "error"
        ])
        # Write data rows
        for result in validation_results:
            writer.writerow([
                result.get("file"),
                result.get("C1", ""),
                ", ".join(result.get("C1_violating_nodes", [])),
                result.get("C2", ""),
                ", ".join(result.get("C2_violating_nodes", [])),
                result.get("C3", ""),
                ", ".join(result.get("C3_violating_nodes", [])),
                result.get("C4", ""),
                ", ".join(result.get("C4_violating_nodes", [])),
                result.get("C5", ""),
                ", ".join(result.get("C5_violating_nodes", [])),
                result.get("C6", ""),
                ", ".join(result.get("C6_violating_nodes", [])),
                result.get("error", ""),
            ])


if __name__ == "__main__":
    variants = ["Public-Baseline", "Public-LADEX-ALG-LLM", "Public-LADEX-ALG-NA", "Public-LADEX-LLM-LLM","Public-LADEX-LLM-NA"]
    llms = ["gpt41mini-","O4Mini-","local-"]

    for v in variants:
        for l in llms:
            for i in range(1, 6):
                gen_dir = f"../Results/{v}/{l}{i}"
                validate_directory(gen_dir)