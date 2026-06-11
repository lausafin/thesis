# Evaluation data

Raw evaluation inputs for [`scripts/reproduce.py`](../scripts/reproduce.py) and [`scripts/eval/`](../scripts/eval/).

## Clone with Git LFS

Large artifacts are stored with Git LFS (~922MB total: ~688MB eval CSVs + ~234MB generations):

```bash
git lfs install
git clone https://github.com/lausafin/thesis
cd thesis
git lfs pull
```

## Layout

| Path | Role |
|------|------|
| `eval/eval_pass1.csv` | Pass-1 screening (LFS) |
| `eval/eval_pass2.csv` | Pass-2 severity scores (LFS) |
| `eval/significance_tests/significance_results.csv` | Precomputed Wilcoxon / BH tests |
| `generations/` | Steered outputs for 36 datasets (LFS): `*_generations.json`, `*_steerability.json` (layer metadata), vectors, plots |
| `human_validation/ratings/rater_a.csv` | Author-rater protocol scores |
| `iaa/agreement_metrics.csv` | Inter-annotator agreement metrics (IAA subsample) |
| `iaa/eval_pass1.csv`, `iaa/eval_pass2.csv` | IAA subsample judge outputs, 1,800 samples × 3 judges (LFS) |
| `mwe_datasets/*.jsonl` | MWE prompts for Stage 1 generation (not committed; see below) |

### MWE prompts (Stage 1 only)

Clone [anthropics/evals](https://github.com/anthropics/evals) (CC-BY-4.0) and copy JSONL files into `data/mwe_datasets/` using the flat filenames in [`experiment/generation/configs/run_all_datasets.sh`](../experiment/generation/configs/run_all_datasets.sh). Source files live under `persona/`, `advanced-ai-risk/`, and `sycophancy/` in that repository.

See [`../REPRODUCTION.md`](../REPRODUCTION.md) for the full workflow.
