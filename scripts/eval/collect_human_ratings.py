#!/usr/bin/env python3
"""Validate and merge human rating CSVs from both raters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from human_eval_common import (  # noqa: E402
    HUMAN_EVAL_DIR,
    PRIMARY_DIMENSIONS,
    parse_sample_id,
)

RATER_FILES = {
    "a": "rater_a.csv",
    "b": "rater_b.csv",
}

REQUIRED_COLS = ["sample_id", "rater_id", "split"] + PRIMARY_DIMENSIONS


def _row_is_complete(row: pd.Series) -> bool:
    for dim in PRIMARY_DIMENSIONS:
        val = row.get(dim)
        if pd.isna(val) or str(val).strip() == "":
            return False
    return True


def _load_rater(path: Path, expected_rater: str) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ratings file: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["rater_id"] = df["rater_id"].astype(str).str.lower()
    bad = df[df["rater_id"] != expected_rater]
    if not bad.empty:
        raise ValueError(f"{path}: rater_id must be '{expected_rater}' for all rows")

    complete_mask = df.apply(_row_is_complete, axis=1)
    incomplete = int((~complete_mask).sum())
    df = df[complete_mask].copy()
    warnings: list[str] = []
    if incomplete:
        warnings.append(
            f"{path.name}: skipped {incomplete} incomplete row(s); only fully scored rows are merged"
        )

    for dim in PRIMARY_DIMENSIONS:
        scores = pd.to_numeric(df[dim], errors="coerce")
        if scores.isna().any():
            raise ValueError(f"{path.name}: non-numeric scores in {dim} after filtering complete rows")
        if ((scores < 0) | (scores > 5)).any():
            raise ValueError(f"{path.name}: scores for {dim} must be 0--5")
        df[dim] = scores.astype(int)
    return df, warnings


def merge_ratings(ratings_dir: Path, split: str) -> tuple[pd.DataFrame, list[str]]:
    parts = []
    warnings: list[str] = []
    for rater, fname in RATER_FILES.items():
        path = ratings_dir / fname
        if not path.exists():
            continue
        df, load_warnings = _load_rater(path, rater)
        warnings.extend(load_warnings)
        if split != "all":
            df = df[df["split"] == split]
        if not df.empty:
            parts.append(df)

    if not parts:
        raise FileNotFoundError(f"No complete rater ratings found in {ratings_dir}")

    merged = pd.concat(parts, ignore_index=True)
    return merged, warnings


def flag_discordant(merged: pd.DataFrame, threshold: int = 2) -> pd.DataFrame:
    rows = []
    for sample_id, grp in merged.groupby("sample_id"):
        if len(grp) != 2:
            rows.append({
                "sample_id": sample_id,
                "split": grp["split"].iloc[0],
                "n_raters": len(grp),
                "discordant": True,
                "max_delta": None,
                "discordant_dims": "incomplete_pair",
            })
            continue
        a = grp.iloc[0]
        b = grp.iloc[1]
        discord_dims = []
        max_delta = 0
        for dim in PRIMARY_DIMENSIONS:
            delta = abs(int(a[dim]) - int(b[dim]))
            max_delta = max(max_delta, delta)
            if delta >= threshold:
                discord_dims.append(dim)
        rows.append({
            "sample_id": sample_id,
            "split": a["split"],
            "n_raters": 2,
            "discordant": len(discord_dims) > 0,
            "max_delta": max_delta,
            "discordant_dims": ",".join(discord_dims) if discord_dims else "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and validate human ratings")
    parser.add_argument(
        "--ratings-dir",
        type=Path,
        default=HUMAN_EVAL_DIR / "ratings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HUMAN_EVAL_DIR,
    )
    parser.add_argument(
        "--split",
        choices=["calibration", "main", "holdout", "all"],
        default="all",
    )
    parser.add_argument("--discordance-threshold", type=int, default=2)
    args = parser.parse_args()

    merged, warnings = merge_ratings(args.ratings_dir, args.split)
    out_name = "human_ratings_merged.csv" if args.split == "all" else f"human_ratings_{args.split}.csv"
    merged_path = args.output_dir / out_name
    merged.to_csv(merged_path, index=False)

    discord = flag_discordant(merged, args.discordance_threshold)
    discord_path = args.output_dir / (
        "human_discordance.csv" if args.split == "all" else f"human_discordance_{args.split}.csv"
    )
    discord.to_csv(discord_path, index=False)

    print(f"Merged {len(merged)} complete ratings -> {merged_path}")
    print(f"Discordant samples: {int(discord['discordant'].sum())} -> {discord_path}")
    for w in warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
