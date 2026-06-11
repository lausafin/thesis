# Full experiment pipeline

Optional upstream stages before [`REPRODUCTION.md`](../REPRODUCTION.md) Stage 3 (`python scripts/reproduce.py`). Archived evaluation CSVs in `data/eval/` and steered generations in `data/generations/` (Git LFS) allow skipping Stages 1–2.

**Source provenance** (from `lausafin/steering-experiments` before archival):

| Component | Branch | Commit |
|-----------|--------|--------|
| CAA generation | `main` | `2a60dca967d1e218ff7e199474455d13f2e5aaed` |
| DeepSeek annotation | `thesis2` | `eb77d9285aa955c57e4adc1df7e14b082f9bd3d2` |

## Prerequisites

- NVIDIA GPU (experiments used A100; ~57 GPU-hours for 31 datasets)
- Hugging Face access to `meta-llama/Llama-3.1-8B-Instruct` (`HF_TOKEN` in `.env`)
- DeepSeek API key (`DEEPSEEK_API_KEY` in `.env`)
- MWE JSONL under `data/mwe_datasets/` for Stage 1 only: clone [anthropics/evals](https://github.com/anthropics/evals) (CC-BY-4.0) and copy files into flat `data/mwe_datasets/<name>.jsonl` (filenames match [`generation/configs/run_all_datasets.sh`](generation/configs/run_all_datasets.sh); sources are under `persona/`, `advanced-ai-risk/`, `sycophancy/`)

Archived steered outputs for all 36 datasets ship in `data/generations/` (Git LFS) for CPU-only Stage 2 annotation and Stage 3 perplexity figures.

```bash
cp .env.example .env   # fill HF_TOKEN and DEEPSEEK_API_KEY
pip install -r experiment/requirements.txt
pip install -r scripts/requirements.txt
```

## Stage 1 — CAA generation (GPU)

From repository root:

```bash
cd experiment/generation
python steering_experiment.py \
  --dataset ../../data/mwe_datasets/power-seeking-inclination.jsonl \
  --max_new_tokens 100 \
  --layers_to_test 10 11 12 13 14 15 16 \
  --output ../../data/generations/power-seeking-inclination/steering_run
```

SLURM array over all datasets:

```bash
cd experiment/generation
sbatch configs/run_all_datasets.sh
```

Outputs: `data/generations/<dataset>/*_generations.json` (plus steering vectors and plots).

## Stage 2 — DeepSeek annotation (API)

From repository root (`PYTHONPATH` set automatically by the script):

```bash
python scripts/eval/run_annotations.py \
  --input data/generations \
  --num_evals 1 \
  --template_path prompts/prompt_pass1.txt
```

Pass 2 runs automatically when `--run_both_passes` is enabled (default). Outputs land in `results/full-run-by-shivam/<timestamp>/eval_pass1.csv` and `eval_pass2.csv`.

Merge multiple batches into `data/eval/`:

```bash
python scripts/concat_eval_batches.py \
  results/full-run-by-shivam/batch1 \
  results/full-run-by-shivam/batch2 \
  results/full-run-by-shivam/batch3
```

Then rerun Wilcoxon tests if needed:

```bash
python scripts/eval/statistical_tests.py \
  data/eval/eval_pass2.csv \
  data/eval/significance_tests/
```

## Stage 3 — Thesis tables and figures

See [`REPRODUCTION.md`](../REPRODUCTION.md) quick start (`python scripts/reproduce.py`).

## Layout

| Path | Role |
|------|------|
| [`generation/`](generation/) | CAA vector extraction and steered generation (`steering_experiment.py`) |
| [`generation/configs/run_all_datasets.sh`](generation/configs/run_all_datasets.sh) | SLURM array job (thesis hyperparameters) |
| [`../scripts/eval/run_annotations.py`](../scripts/eval/run_annotations.py) | Pass-1 / Pass-2 LLM judge |
| [`../prompts/`](../prompts/) | Judge templates |
| [`../core/utils.py`](../core/utils.py), [`../pipeline/judge_parser.py`](../pipeline/judge_parser.py) | Annotation dependencies |
