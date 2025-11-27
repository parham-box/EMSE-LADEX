import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from adjustText import adjust_text
import itertools

BASE_RESULTS_DIR = Path("../Results")
llms = ["gpt41mini-", "O4Mini-", "local-"]
threshold = "no"

# Column Mapping
col_x_name = "coverage_1_to_2"              # L-Match
col_y_name = f"Coverage1 (thr {threshold})" # B-Match


variant_mapping = {
    "Industry-Baseline": "Baseline",
    "Industry-LADEX-ALG-NA": "LADEX-ALG-NA",
    "Industry-LADEX-LLM-NA": "LADEX-LLM-NA",
    "Industry-LADEX-LLM-LLM": "LADEX-LLM-LLM",
    "Industry-LADEX-ALG-LLM": "LADEX-ALG-LLM",
    "Public-Baseline": "Baseline",
    "Public-LADEX-ALG-NA": "LADEX-ALG-NA",
    "Public-LADEX-LLM-NA": "LADEX-LLM-NA",
    "Public-LADEX-LLM-LLM": "LADEX-LLM-LLM",
    "Public-LADEX-ALG-LLM": "LADEX-ALG-LLM"
}


colors = {"Public": "tab:blue", "Industry": "tab:orange"}
labels = {"Public": "Paged", "Industry": "Industry"}


def get_aggregated_data():
    completeness_all = []
    correctness_all = []

    for v_name in variant_mapping.keys():
        
        if not (BASE_RESULTS_DIR / v_name).exists():
            print(f"Warning: Directory {v_name} not found in {BASE_RESULTS_DIR}. Skipping.")
            continue

        for l in llms:
            for i in range(1, 6):
                base_dir = BASE_RESULTS_DIR / v_name / f"{l}{i}"
                llm_dir = base_dir / "llm-as-judge-results"
                auto_dir = base_dir / "results"

                if not llm_dir.exists() or not auto_dir.exists():
                    continue

                gen_name = f"{l}{i}_evaluation_results_gen.csv"
                gt_name  = f"{l}{i}_evaluation_results_gt.csv"

                files = {
                    "Completeness": (llm_dir / "llm_judge_results_gt_to_gen.csv", auto_dir / gt_name),
                    "Correctness":  (llm_dir / "llm_judge_results_gen_to_gt.csv", auto_dir / gen_name),
                }

                for metric, (llm_path, auto_path) in files.items():
                    if not llm_path.exists() or not auto_path.exists():
                        continue

                    try:
                        df_llm = pd.read_csv(llm_path)
                        df_auto = pd.read_csv(auto_path)

                        df_llm.columns = [c.strip() for c in df_llm.columns]
                        df_auto.columns = [c.strip() for c in df_auto.columns]
                        df_llm.rename(columns={"file": "File ID"}, inplace=True)

                        df_llm["File ID"] = df_llm["File ID"].astype(str).str.replace(".txt", "", regex=False)
                        df_auto["File ID"] = df_auto["File ID"].astype(str)

                        if col_y_name not in df_auto.columns:
                            continue

                        merged = pd.merge(
                            df_llm[["File ID", col_x_name]],
                            df_auto[["File ID", col_y_name]],
                            on="File ID",
                            how="inner"
                        )
                        
                        merged["variant"] = v_name
                        merged["llm"] = l
                        merged["run"] = i

                        if metric == "Completeness":
                            completeness_all.append(merged)
                        else:
                            correctness_all.append(merged)
                    except Exception as e:
                        print(f"Error processing {llm_path}: {e}")

    results = {}
    for metric_name, data_list in [("Completeness", completeness_all), ("Correctness", correctness_all)]:
        if not data_list:
            print(f"No data found for {metric_name}")
            results[metric_name] = pd.DataFrame()
            continue

        df = pd.concat(data_list, ignore_index=True)

        df[col_x_name] = pd.to_numeric(df[col_x_name], errors='coerce')
        df[col_y_name] = pd.to_numeric(df[col_y_name], errors='coerce')
        
        df.dropna(subset=[col_x_name, col_y_name], inplace=True)

        df_agg = df.groupby("variant")[[col_x_name, col_y_name]].mean().reset_index()
        results[metric_name] = df_agg
        
        print(f"Processed {len(df)} rows for {metric_name}, aggregated to {len(df_agg)} variants.")

    return results

def transform_data_for_plotting(results_dict):
    plot_data = {
        "Public": {"Completeness": {}, "Correctness": {}},
        "Industry": {"Completeness": {}, "Correctness": {}}
    }

    for metric, df_agg in results_dict.items():
        if df_agg.empty: 
            continue
            
        for _, row in df_agg.iterrows():
            variant = row['variant']
            val_x = row[col_x_name]
            val_y = row[col_y_name]

            if "Public" in variant:
                group = "Public"
            elif "Industry" in variant:
                group = "Industry"
            else:
                continue 

            plot_data[group][metric][variant] = [val_x, val_y]
            
    return plot_data

def plot_with_separate_trends(score_type, filename, plot_data):
    print(f"\nGeneratng plot: {filename}...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    texts = []
    
    group_data = {"Public": {"x": [], "y": []}, "Industry": {"x": [], "y": []}}
    
    has_data = False
    
    for group in ["Public", "Industry"]:
        if score_type not in plot_data[group] or not plot_data[group][score_type]:
            continue

        for variant, values in plot_data[group][score_type].items():
            has_data = True
            short_name = variant_mapping.get(variant, variant)
            x, y = values
            
            group_data[group]["x"].append(x)
            group_data[group]["y"].append(y)
            
            ax.scatter(x, y, color=colors[group], s=150, zorder=3, 
                       edgecolors='white', linewidth=1.5, alpha=0.8)
            
            import random
            offset_x = (random.random() - 0.5) * 0.01
            offset_y = (random.random() - 0.5) * 0.01

            txt = ax.text(
                x + offset_x, y + offset_y, short_name,
                fontsize=14,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[group],
                        alpha=0.3, edgecolor='none')
            )

            texts.append(txt)
    
    if not has_data:
        print(f"Skipping plot {filename} (No data found)")
        plt.close()
        return

    for group in ["Public", "Industry"]:
        x_data = np.array(group_data[group]["x"])
        y_data = np.array(group_data[group]["y"])
        
        if len(x_data) < 2: continue

        coeffs = np.polyfit(x_data, y_data, 1)
        slope, intercept = coeffs[0], coeffs[1]
        
        spearman_corr, spearman_p = spearmanr(x_data, y_data)
        
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        y_line = slope * x_line + intercept
        
        ax.plot(x_line, y_line, color=colors[group], linestyle='--', linewidth=2, alpha=0.6)

        ax.scatter([], [], color=colors[group], s=150, 
                   label=f'{labels[group]} (Spearman Correlation={spearman_corr:.2f}, p-value={spearman_p:.2f})')
    
    all_x = group_data["Public"]["x"] + group_data["Industry"]["x"]
    all_y = group_data["Public"]["y"] + group_data["Industry"]["y"]
    
    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_range, y_range = x_max - x_min, y_max - y_min
        
        padding = 0.25
        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    
        adjust_text(
            texts,
            x=all_x,
            y=all_y,
            arrowprops=dict(arrowstyle='-', color='black', lw=0.7, alpha=0.6),
            ax=ax,
            expand_points=(5, 5),
            expand_text=(2.0, 2.0),
            force_points=(0.3, 0.3),
            force_text=(0.7, 0.7),
            only_move={'points': 'xy', 'text': 'xy'},
            autoalign='y',
            lim=3000,
            avoid_points=True
        )
    
    ax.set_xlabel("L-Match", fontsize=16, fontweight='bold')
    ax.set_ylabel("B-Match", fontsize=16, fontweight='bold')
    ax.set_title(f"Average {score_type} Scores", 
                 fontsize=16, fontweight='bold')
    
    ax.legend(loc='upper left', frameon=True, fontsize=16, prop={'weight': 'bold', 'size':'13'})
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16, width=1.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Saved {filename}")
    plt.close()

if __name__ == "__main__":
    results_df = get_aggregated_data()
    
    plot_data_structure = transform_data_for_plotting(results_df)
    
    plot_with_separate_trends("Completeness", "rq1_completeness_scores.pdf", plot_data_structure)
    plot_with_separate_trends("Correctness", "rq1_correctness_scores.pdf", plot_data_structure)