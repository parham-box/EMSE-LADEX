import os
import csv

def process_single_run_violations(run_directory):
    run_violations = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0, "C6": 0}
    diagrams_with_any_violation = set()
    total_rows = 0

    constraint_file_path = os.path.join(run_directory, "structural_constraints.csv")
    
    if not os.path.exists(constraint_file_path):
        print(f"Warning: Could not find structural_constraints.csv in {run_directory}")
        return run_violations, 0, 0

    with open(constraint_file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_rows += 1
            file_id = row.get("file")
            violated = False
            for constraint in run_violations.keys():
                if row.get(constraint) == "False":
                    run_violations[constraint] += 1
                    violated = True
            if violated:
                diagrams_with_any_violation.add(file_id)

    return run_violations, len(diagrams_with_any_violation), total_rows

if __name__ == "__main__":
    approach = "Public-LADEX-LLM-NA"
    model_paths = ["gpt41mini-", "O4Mini-","local-"]
    base_dir = "Results"

    final_summary = {}

    for path_prefix in model_paths:
        print(f"--- Processing model: {path_prefix} ---")

        final_summary[path_prefix] = {
            "C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0, "C6": 0,
            "Diagrams_with_violation": 0,
            "Total_diagrams": 0
        }

        for i in range(1, 6):
            run_dir = os.path.join(base_dir, approach, f"{path_prefix}{i}","results")
            run_violations, num_with_violation, total_diagrams = process_single_run_violations(run_dir)

            for constraint, count in run_violations.items():
                final_summary[path_prefix][constraint] += count

            final_summary[path_prefix]["Diagrams_with_violation"] += num_with_violation
            final_summary[path_prefix]["Total_diagrams"] += total_diagrams

    # Print the final summary
    print("\n--- Final Aggregated Summary ---")
    for model, violations in final_summary.items():
        print(f"\nModel: {model}")
        total = violations["Total_diagrams"]
        num_with_violation = violations["Diagrams_with_violation"]
        percent_with_violation = (num_with_violation / total * 100) if total > 0 else 0.0

        total_constraint_violations = sum(v for k, v in violations.items() if k.startswith("C"))
        percent_total_violations = (total_constraint_violations / total * 100) if total > 0 else 0.0

        print(f"  Total Diagrams: {total}")
        print(f"  Total Violations (across constraints): {total_constraint_violations} ({percent_total_violations:.2f}%)")
        print(f"  Unique Diagrams with Any Violation: {num_with_violation} ({percent_with_violation:.2f}%)")

        for constraint in ["C1", "C2", "C3", "C4", "C5", "C6"]:
            count = violations[constraint]
            percent = (count / total * 100) if total > 0 else 0.0
            print(f"    {constraint}: {count} ({percent:.2f}%)")

    # Save results to CSV inside the approach folder
    output_dir = os.path.join(base_dir, approach)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "structural_results.csv")

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Model", "Total_diagrams", "Diagrams_with_violation",
                      "Total_violations", "C1", "C2", "C3", "C4", "C5", "C6"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model, violations in final_summary.items():
            total = violations["Total_diagrams"]
            total_constraint_violations = sum(v for k, v in violations.items() if k.startswith("C"))
            row = {
                "Model": model,
                "Total_diagrams": total,
                "Diagrams_with_violation": violations["Diagrams_with_violation"],
                "Total_violations": total_constraint_violations,
                "C1": violations["C1"],
                "C2": violations["C2"],
                "C3": violations["C3"],
                "C4": violations["C4"],
                "C5": violations["C5"],
                "C6": violations["C6"],
            }
            writer.writerow(row)

    print(f"\nResults written to {output_file}")
