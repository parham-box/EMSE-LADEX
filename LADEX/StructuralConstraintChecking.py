class StructuralConstraintChecking:
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

