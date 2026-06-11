#!/usr/bin/env python3
"""Merge Pass-1/Pass-2 CSVs from multiple annotation batches into data/eval/."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

THESIS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = THESIS_ROOT / "data" / "eval"


def concat_batches(batch_dirs: list[Path], output_dir: Path) -> None:
    pass1_dfs = []
    pass2_dfs = []
    for batch in batch_dirs:
        pass1_path = batch / "eval_pass1.csv"
        pass2_path = batch / "eval_pass2.csv"
        if not pass1_path.exists() or not pass2_path.exists():
            raise FileNotFoundError(f"Missing eval_pass1.csv or eval_pass2.csv in {batch}")
        pass1_dfs.append(pd.read_csv(pass1_path))
        pass2_dfs.append(pd.read_csv(pass2_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_pass1 = pd.concat(pass1_dfs, ignore_index=True)
    combined_pass2 = pd.concat(pass2_dfs, ignore_index=True)
    combined_pass1.to_csv(output_dir / "eval_pass1.csv", index=False)
    combined_pass2.to_csv(output_dir / "eval_pass2.csv", index=False)
    print(f"Wrote {output_dir / 'eval_pass1.csv'} ({combined_pass1.shape})")
    print(f"Wrote {output_dir / 'eval_pass2.csv'} ({combined_pass2.shape})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate eval CSV batches into data/eval/")
    parser.add_argument(
        "batches",
        nargs="+",
        type=Path,
        help="Batch directories containing eval_pass1.csv and eval_pass2.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()
    concat_batches([b.resolve() for b in args.batches], args.output_dir.resolve())


if __name__ == "__main__":
    main()
