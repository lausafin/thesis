"""Path resolution for thesis result regeneration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = THESIS_ROOT / "data"
DEFAULT_OUTPUT_DIR = THESIS_ROOT / "results"


@dataclass(frozen=True)
class DataPaths:
    data_root: Path
    combined_dir: Path
    pass1_csv: Path
    pass2_csv: Path
    significance_csv: Path
    generations_dir: Path
    iaa_agreement_csv: Path
    iaa_pass1_csv: Path
    iaa_pass2_csv: Path
    output_dir: Path


def resolve_data_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env_root = os.environ.get("STEERING_EXPERIMENTS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return DATA_DIR


def resolve_paths(
    data_root: Path | None = None,
    output_dir: Path | None = None,
) -> DataPaths:
    root = resolve_data_root(data_root)
    out = (output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    return DataPaths(
        data_root=root,
        combined_dir=root / "eval",
        pass1_csv=root / "eval/eval_pass1.csv",
        pass2_csv=root / "eval/eval_pass2.csv",
        significance_csv=root / "eval/significance_tests/significance_results.csv",
        generations_dir=root / "generations",
        iaa_agreement_csv=root / "iaa/agreement_metrics.csv",
        iaa_pass1_csv=root / "iaa/eval_pass1.csv",
        iaa_pass2_csv=root / "iaa/eval_pass2.csv",
        output_dir=out,
    )
