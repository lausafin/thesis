# Reproduction Guide

Two paths:

1. **Quick start (archived data)** — regenerate thesis tables and figures from committed evaluation CSVs and generations (CPU only; `git lfs pull` required).
2. **Full pipeline (optional, GPU + API)** — rerun CAA generation and DeepSeek judging; see [`experiment/README.md`](experiment/README.md).

Judge templates live in [`prompts/`](prompts/). Evaluation scripts live in [`scripts/eval/`](scripts/eval/). Raw evaluation CSVs and steered generations live in [`data/`](data/) (Git LFS; `git lfs pull` required).

For modular flags and package internals, see [`scripts/README.md`](scripts/README.md). For run provenance, see [`results/MANIFEST.md`](results/MANIFEST.md).

## Quick start (archived data)

From the repository root (`git lfs pull` fetches eval CSVs and generation JSON):

```bash
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r scripts/requirements.txt
python scripts/reproduce.py
```

Outputs land in `results/` (CSVs under `results/tables/`, PNGs under `results/figures/`, qualitative examples under `results/examples/`).

## Full pipeline (optional)

Requires GPU, Hugging Face model access, and DeepSeek API credentials. Copy [`.env.example`](.env.example) to `.env` and install [`experiment/requirements.txt`](experiment/requirements.txt) in addition to `scripts/requirements.txt`.

```bash
# Stage 1 — GPU (see experiment/generation/configs/run_all_datasets.sh)
cd experiment/generation && sbatch configs/run_all_datasets.sh

# Stage 2 — API
python scripts/eval/run_annotations.py \
  --input data/generations \
  --num_evals 1 \
  --template_path prompts/prompt_pass1.txt

# If multiple annotation batches:
python scripts/concat_eval_batches.py results/full-run-by-shivam/batch1 ...

# Stage 3 — CPU (same as quick start)
python scripts/reproduce.py
```

Details: [`experiment/README.md`](experiment/README.md).

## Inputs

`scripts/reproduce.py` reads evaluation CSVs from [`data/`](data/) in this repository:

| File | Role |
|------|------|
| `data/eval/eval_pass1.csv` | Pass-1 screening (**Git LFS**) |
| `data/eval/eval_pass2.csv` | Pass-2 severity scores (**Git LFS**) |
| `data/eval/significance_tests/significance_results.csv` | Precomputed Wilcoxon / BH tests |
| `data/iaa/agreement_metrics.csv` | Inter-annotator agreement metrics |
| `data/iaa/eval_pass1.csv`, `data/iaa/eval_pass2.csv` | IAA subsample judge outputs (**Git LFS**) |
| `data/generations/*/*` | Steered outputs for 36 datasets (**Git LFS**); perplexity figures and optional Stage 2 rerun |

Optional:

| Path | Role |
|------|------|
| `data/mwe_datasets/*.jsonl` | MWE prompts for Stage 1; clone [anthropics/evals](https://github.com/anthropics/evals) (see [`data/README.md`](data/README.md)) |

Override the data root for testing with `--data-root` (must follow the same `data/` layout).

## Outputs

`reproduce.py` regenerates the curated bundle consumed by the thesis:

- **`results/tables/`** — summary CSVs (significance tiers, funnel counts, human-validation tables if present, etc.)
- **`results/figures/`** — PNG figures referenced via `\includegraphics` in `thesis/sections/`
- **`results/examples/`** — qualitative case-study markdown for appendices

## Modular runs

Pass through the same flags as `regenerate_results.py` (alias: `python -m scripts.thesis_results`):

```bash
# Figures only (after tables exist)
python scripts/reproduce.py --skip-tables --skip-examples --skip-iaa --skip-pass1-pass2 --skip-per-alpha --skip-perplexity

# Perplexity analysis only (requires data/generations/)
python scripts/reproduce.py --skip-tables --skip-figures --skip-examples --skip-iaa --skip-pass1-pass2 --skip-per-alpha

# Per-α objective metrics only
python scripts/reproduce.py --skip-tables --skip-figures --skip-examples --skip-iaa --skip-pass1-pass2 --skip-perplexity
```

Full flag list: `python scripts/reproduce.py --help`.

## Statistical significance tests

Wilcoxon signed-rank tests with Benjamini–Hochberg correction can be rerun from this repository:

```bash
python scripts/eval/statistical_tests.py \
  data/eval/eval_pass2.csv \
  data/eval/significance_tests/
```

Then run `python scripts/reproduce.py`. See [`results/MANIFEST.md`](results/MANIFEST.md) for batch provenance.

## LaTeX table fragments

Nine files under `thesis/sections/generated/` are **committed LaTeX snapshots**. `reproduce.py` does not rewrite them automatically; after regenerating CSVs, update these files manually (or via the human-validation pipeline below) so numbers match `results/tables/`.

| LaTeX fragment | Included from | Primary CSV / source |
|----------------|---------------|----------------------|
| `primary_claims_summary.tex` | `thesis/sections/results.tex` | `significance_primary_claims.csv`, tier summaries |
| `pass1_pass2_funnel.tex` | `thesis/sections/results.tex` | `pass1_pass2_funnel.csv` |
| `dimension_claim_summary.tex` | `thesis/sections/results.tex` | `dimension_claim_summary.csv` |
| `practical_significance_sensitivity.tex` | `thesis/sections/results.tex` | `practical_significance_sensitivity.csv` |
| `primary_claim_criteria.tex` | `thesis/sections/methodology.tex` | Pre-specified thresholds (Methods); sync with `practical_significance.py` constants |
| `human_validation_concordance.tex` | `thesis/sections/appendix_human_validation.tex` | Hand-maintained; informed by `human_validation_concordance_main.csv` |
| `human_validation_holdout_screen.tex` | `thesis/sections/appendix_human_validation.tex` | Hand-maintained; informed by holdout CSVs |
| `human_validation_exploratory_concordance.tex` | (committed; optional) | `analyze_human_validation.py --write-tex exploratory-concordance` |
| `human_validation_iaa.tex` | (committed; optional) | Human-validation IAA export scripts |

Figure captions and prose remain in `thesis/sections/*.tex`; embedded plot titles are set in `scripts/thesis_results/*.py`.

## Human validation

Ratings: `data/human_validation/ratings/rater_a.csv`. Scripts: `scripts/eval/`.

```bash
pip install krippendorff
python scripts/eval/collect_human_ratings.py --split all

python scripts/eval/analyze_human_validation.py \
  --split main --single-rater a --write-tex exploratory-concordance \
  --thesis-generated thesis/sections/generated \
  --thesis-tables results/tables
python scripts/eval/analyze_human_validation.py \
  --split holdout --single-rater a \
  --thesis-generated thesis/sections/generated \
  --thesis-tables results/tables
python scripts/eval/analyze_human_validation.py \
  --split calibration --single-rater a \
  --thesis-generated thesis/sections/generated \
  --thesis-tables results/tables
```

The export script writes `human_validation_exploratory_concordance.tex`; the thesis uses hand-maintained `human_validation_concordance.tex` and `human_validation_holdout_screen.tex`. Key CSV outputs: `human_validation_concordance_main.csv`, `human_validation_logit_audit_main.csv`, `human_validation_concordance_holdout.csv`.

Subsampling for the protocol: `scripts/eval/prepare_human_eval_subsample.py`.

## Compile the PDF

Figures load from `results/figures/` via `\graphicspath{{../results/figures/}}` in `thesis/main.tex`. Run `python scripts/reproduce.py` first, then:

```bash
cd thesis && latexmk -pdf main.tex
```

## Limitations

- **Analysis-only path:** archived Pass-1/Pass-2 CSVs, generation JSON (`data/generations/`), and `python scripts/reproduce.py` reproduce tables and figures without GPU or API access.
- **Full pipeline:** Stage 1 (GPU) and Stage 2 (API) scripts are included; rerunning them requires Hugging Face model access, DeepSeek API credits, and MWE JSONL from [anthropics/evals](https://github.com/anthropics/evals) (not vendored in this repo).
- Wilcoxon / BH tests are consumed by default; rerun via `scripts/eval/statistical_tests.py` if needed.
- LaTeX table fragments require manual sync unless regenerated via the human-validation export commands above.
