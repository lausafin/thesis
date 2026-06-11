# Run Provenance (Full Run)

* **Batches:** `20260518_184427`, `20260519_021515`, `20260519_110953`
* **Inputs:** `data/batch1`, `data/batch2`, `data/batch3`
* **Flags:** `--num_evals 1`, two-pass (Pass 1 then Pass 2 on flagged)
* **Pass 1 samples:** 28,807
* **Pass 1 flagged (Pass 2 intended):** 16,021 (55.6% of corpus)
* **Pass 2 export (parseable severity):** 16,015 (six flagged rows lack complete Pass-2 judge output; see `tables/run_summary.csv`)

## Regeneration

Statistical tests (from repository root):

```bash
python scripts/eval/statistical_tests.py \
  data/eval/eval_pass2.csv \
  data/eval/significance_tests/
```

Thesis export bundle:

```bash
pip install -r scripts/requirements.txt
python scripts/reproduce.py
```

Perplexity vs side-effect correlations require `data/generations/*/*_generations.json` (optional; use `--skip-perplexity` if absent). See [`scripts/README.md`](../scripts/README.md).

Output directory: `results/`

## Human validation (single author-rater)

Ratings live in `data/human_validation/ratings/rater_a.csv`.
The author-rater completed 85/85 protocol samples (20 calibration, 30 main, 35 holdout).

Regenerate CSV tables and LaTeX fragments:

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

Generated TeX: script outputs `human_validation_exploratory_concordance.tex`; thesis uses hand-maintained `thesis/sections/generated/human_validation_concordance.tex` and `human_validation_holdout_screen.tex`.

Key outputs: `tables/human_validation_concordance_main.csv`, `tables/human_validation_logit_audit_main.csv`, `tables/human_validation_concordance_holdout.csv`.
