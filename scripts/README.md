# Thesis Results Regeneration

Self-contained Python package for reproducing `results/tables/` and `results/figures/` from evaluation CSVs.

**Entry point:** [`REPRODUCTION.md`](../REPRODUCTION.md) at repo root; run `python scripts/reproduce.py` (equivalent to `regenerate_results.py` below).

## Setup

From the thesis repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

## Data paths

Evaluation CSVs live in [`data/`](../data/) (Pass-1/Pass-2 via Git LFS; see [`data/README.md`](../data/README.md)):

```
data/eval/eval_pass1.csv
data/eval/eval_pass2.csv
data/eval/significance_tests/significance_results.csv
data/generations/   # perplexity JSON (optional)
```

Override with `--data-root` for testing (same layout under the given root).

## Regenerate everything

```bash
python scripts/regenerate_results.py
# or
python -m scripts.thesis_results
```

Outputs land in `results/` (tables, figures, examples, notes).

## Modular runs

```bash
# Figures only (after tables exist for run_summary.csv)
python scripts/regenerate_results.py --skip-tables --skip-examples --skip-iaa --skip-pass1-pass2 --skip-per-alpha --skip-perplexity

# Perplexity analysis only
python scripts/regenerate_results.py --skip-tables --skip-figures --skip-examples --skip-iaa --skip-pass1-pass2 --skip-per-alpha

# Per-α objective metrics only
python scripts/regenerate_results.py --skip-tables --skip-figures --skip-examples --skip-iaa --skip-pass1-pass2 --skip-perplexity
```

## Package layout

| Module | Role |
|--------|------|
| `config.py` | Path resolution (`STEERING_EXPERIMENTS_ROOT`, defaults) |
| `constants.py` | Pass-1/Pass-2 schema constants |
| `tables.py` | Summary CSVs, practical-significance tiers |
| `figures.py` | Heatmaps, bar charts, severity histograms |
| `pass1_pass2.py` | Screening funnel, concordance, boxplots |
| `per_alpha.py` | Token-length / logit per-α curves |
| `perplexity.py` | Perplexity vs side-effect correlations |
| `practical_significance.py` | Primary vs exploratory claim tiers |
| `examples.py` | Qualitative case-study markdown exports |
| `export.py` | Orchestrator CLI |

## Plot titles

Embedded figure titles are set in the plotting functions above (`plt.title`, `ax.set_title`, `fig.suptitle`). LaTeX figure captions are edited separately in `thesis/sections/*.tex`.

## Note on statistical tests

Wilcoxon / BH significance tests can be rerun via [`scripts/eval/statistical_tests.py`](eval/statistical_tests.py); `reproduce.py` consumes the precomputed `data/eval/significance_tests/significance_results.csv` by default.
