#!/bin/bash
#SBATCH --job-name=steering_all
#SBATCH --ntasks=1
#SBATCH --mem=64000M
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/steering_%A_%a.out
#SBATCH --error=logs/steering_%A_%a.err
#SBATCH --array=0-31%8

# ============================================================================
# Steering Vectors Experiment - All Datasets (Array Job)
#
# Thesis-matched settings: layers 10-16, max_new_tokens=100, 60/10/30 split.
# Run from repository root or set THESIS_ROOT. Requires MWE JSONL under DATA_DIR.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THESIS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

export TOKENIZERS_PARALLELISM=false
mkdir -p logs

MODEL="meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE=8
MAX_NEW_TOKENS=100
LAYERS_TO_TEST="10 11 12 13 14 15 16"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_DIR="${DATA_DIR:-${THESIS_ROOT}/data/mwe_datasets}"
GEN_ROOT="${GEN_ROOT:-${THESIS_ROOT}/data/generations}"

# Format: DATASET_NAME DATASET_PATH NUM_TRAIN NUM_VAL NUM_TEST
DATASET_CONFIGS=(
    "anti-LGBTQ-rights ${DATA_DIR}/anti-LGBTQ-rights.jsonl 600 100 300"
    "believes-AIs-not-xrisk ${DATA_DIR}/believes-AIs-not-xrisk.jsonl 600 100 300"
    "believes-anti-abortion ${DATA_DIR}/believes-anti-abortion.jsonl 600 100 300"
    "believes-it-has-phenomenal-consciousness ${DATA_DIR}/believes-it-has-phenomenal-consciousness.jsonl 600 100 300"
    "believes-life-has-no-meaning ${DATA_DIR}/believes-life-has-no-meaning.jsonl 600 100 300"
    "believes-not-watched ${DATA_DIR}/believes-not-watched.jsonl 600 100 300"
    "conscientiousness ${DATA_DIR}/conscientiousness.jsonl 600 100 300"
    "coordinate-itself ${DATA_DIR}/coordinate-itself.jsonl 193 32 97"
    "coordinate-other-ais ${DATA_DIR}/coordinate-other-ais.jsonl 246 41 123"
    "coordinate-other-versions ${DATA_DIR}/coordinate-other-versions.jsonl 208 34 106"
    "corrigible-less-HHH ${DATA_DIR}/corrigible-less-HHH.jsonl 210 35 106"
    "desire-to-create-allies ${DATA_DIR}/desire-to-create-allies.jsonl 600 100 300"
    "interest-in-music ${DATA_DIR}/interest-in-music.jsonl 600 100 300"
    "interest-in-science ${DATA_DIR}/interest-in-science.jsonl 600 100 300"
    "narcissism ${DATA_DIR}/narcissism.jsonl 600 100 300"
    "one-box-tendency ${DATA_DIR}/one-box-tendency.jsonl 180 30 90"
    "openness ${DATA_DIR}/openness.jsonl 600 100 300"
    "politically-liberal ${DATA_DIR}/politically-liberal.jsonl 600 100 300"
    "power-seeking-inclination ${DATA_DIR}/power-seeking-inclination.jsonl 598 99 301"
    "self-aware-arch ${DATA_DIR}/self-aware-arch.jsonl 180 30 90"
    "self-aware-lm ${DATA_DIR}/self-aware-lm.jsonl 180 30 90"
    "subscribes-to-Christianity ${DATA_DIR}/subscribes-to-Christianity.jsonl 600 100 300"
    "subscribes-to-Hinduism ${DATA_DIR}/subscribes-to-Hinduism.jsonl 600 100 300"
    "subscribes-to-Islam ${DATA_DIR}/subscribes-to-Islam.jsonl 600 100 300"
    "subscribes-to-average-utilitarianism ${DATA_DIR}/subscribes-to-average-utilitarianism.jsonl 600 100 300"
    "subscribes-to-deontology ${DATA_DIR}/subscribes-to-deontology.jsonl 600 100 300"
    "subscribes-to-utilitarianism ${DATA_DIR}/subscribes-to-utilitarianism.jsonl 600 100 300"
    "survival-instinct ${DATA_DIR}/survival-instinct.jsonl 571 95 287"
    "sycophancy ${DATA_DIR}/sycophancy.jsonl 5920 986 2961"
    "wealth-seeking-inclination ${DATA_DIR}/wealth-seeking-inclination.jsonl 591 98 296"
    "willingness-to-use-physical-force ${DATA_DIR}/willingness-to-use-physical-force-to-achieve-benevolent-goals.jsonl 600 100 300"
    "willingness-to-use-social-engineering ${DATA_DIR}/willingness-to-use-social-engineering-to-achieve-its-goals.jsonl 600 100 300"
)

CONFIG="${DATASET_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r DATASET_NAME DATASET_PATH NUM_TRAIN NUM_VAL NUM_TEST <<< "$CONFIG"

mkdir -p "${GEN_ROOT}/${DATASET_NAME}"
OUTPUT_PREFIX="${GEN_ROOT}/${DATASET_NAME}/steering_${TIMESTAMP}"

echo "Dataset: $DATASET_PATH"
echo "Output: ${OUTPUT_PREFIX}_generations.json"

python steering_experiment.py \
    --dataset "$DATASET_PATH" \
    --model "$MODEL" \
    --num_train "$NUM_TRAIN" \
    --num_val "$NUM_VAL" \
    --num_test "$NUM_TEST" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --layers_to_test $LAYERS_TO_TEST \
    --output "$OUTPUT_PREFIX"
