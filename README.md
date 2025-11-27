# The Impact of Critique on LLM-Based Model Generation from Natural Language: The Case of Activity Diagrams

## Abstract

Large Language Models (LLMs) show strong potential for automating model generation from natural-language descriptions. A common approach begins with an initial model generation, followed by an iterative critique-refine loop in which the model is evaluated for issues and refined based on those issues.  This process needs to address: (1) structural correctness  -- compliance with well-formedness rules  --  and (2) semantic alignment -- accurate reflection of the intended meaning in the source text.  We present LADEX (LLM-based Activity Diagram Extractor), a pipeline for deriving activity diagrams from natural-language process descriptions using an LLM-driven critique-refine process. Structural checks in LADEX can be performed either algorithmically or by an LLM, while alignment checks are performed by an LLM. We design five ablated variants of LADEX to study: (i)~the impact of the critique-refine loop itself, (ii)~the role of LLM-based semantic checks, and (iii)~the comparative effectiveness of algorithmic versus LLM-based structural checks.

To evaluate LADEX, we compare the generated activity diagrams with expert-created ground truths using both trace-based behavioural matcher   and an LLM-based activity-diagram matcher. This enables automated measurement of correctness (whether the generated diagram includes the ground-truth nodes) and completeness (how many of the ground-truth nodes the generated diagram covers). Experiments on two datasets -- a public-domain dataset and an industry dataset from our collaborator, Ciena -- indicate that: (1)~Both the behavioural and LLM matchers yield similar completeness and correctness comparisons across the LADEX variants.
(2)~The critique–refine loop improves structural validity, correctness, and completeness compared to single-pass generation. 
(3)~Activity diagrams refined based on algorithmic structural checks achieve structural consistency, whereas those refined based on LLM-based checks often still show structural inconsistencies. (4)~Combining algorithmic structural checks with LLM-based semantic checks (using O4 Mini) delivers the strongest results -- up to 86% correctness and 92% completeness -- while requiring fewer than five LLM calls on average. In contrast, using only algorithmic structural checks reaches similar correctness (86%) and slightly lower completeness (90%) with just over one LLM call, making it the preferred low-cost option.

## Getting Started

The repository contains the code for all five variants of **LADEX**, located in the `LADEX` directory.
Each variant includes a `call_llm_api` method, which must be updated to invoke an LLM and return the response in the format expected by the code.

By default, `call_llm_api` is set to use an OpenAI implementation, and the model can be selected using the `--model` flag.

To run each variant of LADEX, use:

`python [VARIANT].py --model [MODEL_NAME] --dataset [LOCATION_OF_DATASET] --output [LOCATION_OF_OUTPUT] --verbose True`

The `LADEX` directory also contains `StructuralConstraintChecking.py`, which is used for algorithmic structural checking in the variants that require it.

## Complete Prompts

All prompts used for the different steps of the LADEX pipeline and for different variants are provided in `LADEX/prompts.py`. The prompt for `L-Match` is provided in `Evaluation_Code/LLMMatcher.py` .

## Evaluation Code

### Structural Soundness

* `StructuralConstraintEvaluation.py` defines all of the formal structural constraints (see **Table 1**) and checks if an activity diagram conforms to them.
* By modifying the `__main__` function, you can specify which variants and LLMs to evaluate.
* To count the overall structural correctness of the generated diagrams, first perform per-diagram structural checks, then use `CountStructuralIssues.py`. Modify its `__main__` function to summarize overall structural soundness results.

### Semantic Correctness and Completeness

#### B-Match

* `B-Match` implements the B-Match algorithm for comparing and matching the nodes of two activity diagrams.
* To run it, modify the `__main__` function in `B-Match.py` and set the directories for both sets of activity diagrams.
* Ensure that two corresponding diagrams in each directory share the same name so the implementation can correctly match them.
* B-Match outputs raw evaluations for each file. To summarize results across variants, LLMs, and runs, use `SummarySemanticResults.py` and update its constants to point to the relevant evaluation results.
#### L-Match
* `LLMMatcher` implements the L-Match used to generate a matching between the nodes of two activity diagrams.
* To run it, modify the `__main__` function in `LLMMatcher.py` and set the directories for both sets of activity diagrams, and overload the `call_llm_api` method properly, and update the `set_model` call in main with your prefered LLM.
* Ensure that two corresponding diagrams in each directory share the same name so the implementation can correctly match them.
* L-Match outputs the raw matchings between two diagrams and compute semantic completeness and correctness values based on the matchings. To summarize results across variants, LLMs, and runs, use `SummarySemanticResultsLLMMatcher.py` and update its constants to point to the relevant evaluation results.

### Cost

* `CostEvaluation.py` uses the `_metrics.json` file generated for each generated diagram to compute the average number of LLM calls and tokens used.
* Modify the `__main__` function to select different variants or LLMs.

## Evaluation Results

All evaluation results are located in the `results` directory.

For each variant of LADEX, generated diagram and evaluation results are provided for both the **PAGED** and **Industry** datasets.

* Each variant directory includes outputs from five runs.
* For the PAGED dataset, the results contain the generated diagrams and their cost results are included.
* For the Industry dataset, only the cost results are included.
* Other results include:

  * Overall Semantic correctness and completeness using B-Match `Summary_Integrated.csv`
  * Overall Semantic correctness and completeness using L-Match `Summary_Integrated_llm_matcher.csv`
  * Overall Structural violations `structural_results.csv`
  * Overall Cost `cost_evaluation.csv`


## Dataset

The `Dataset` directory contains:

* For the **PAGED** dataset: the process descriptions and ground-truth activity diagrams used for evaluation.
* For the **Industry** dataset: a subset of anonymized process descriptions and anonymized ground-truth activity diagrams from the Industry dataset.

## L-Match Correlation
* This directory contains the five diagrams for each dataset used to find the correlation of L-Match and expert judgment. `Ground-Truth` and `LLM-generated` contain the ground truth and generated diagrams for `PAGED` dataset. `Human-Matchings` and `L-Match`, respectively, contain the matchings between the two diagrams found by a human expert and L-Match for both `PAGED` and Industry dataset.
* `comp.py` uses standarad recall-precision metrics to compare these results, and stores the output in `final_score.csv`.


## Research Questions (RQ1, RQ2, RQ3, RQ4)

Results for the research questions are located in the `RQ1`, `RQ2`, `RQ3`, and `RQ4` directories.
### RQ1:
* directory `rq1` contains the file `rq1.py` which when executed generates the scatter plots used in RQ1.

### RQ2, RQ3, and RQ4:

* directories for statistcal tests for both `L-Match` and `B-Match`.
* `StatisticalTest.py`, which can be configured to perform statistical tests (Wilcoxon paired test and A12) between the results of two variants based on the LLMs used.
* Results of the statistical tests for all variants used in the respective research question.
* `bh.py` includes the code for computing the correct p-values using Benjamini-Hochberg correction. `All_Stats_with_BH.csv` files include the corrected p-values for all statistical tests.
