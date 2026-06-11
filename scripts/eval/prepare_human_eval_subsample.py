#!/usr/bin/env python3
"""Prepare stratified subsamples for human validation (calibration, main, holdout)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from human_eval_common import (  # noqa: E402
    DEFAULT_PASS1,
    DEFAULT_PASS2,
    GOLD_CASE_QUERIES,
    HUMAN_EVAL_DIR,
    PRIMARY_DIMENSIONS,
    attach_llm_refs,
    find_gold_rows,
    is_pass1_flagged,
    llm_col,
    pass2_all_zero,
    question_text,
    sample_id,
    severity_tertile,
)

N_CALIBRATION = 20
N_MAIN_TARGET = 120
N_PER_DATASET = 3
N_HOLDOUT_PER_DATASET = 1
N_HIGH_ASYMMETRY_EXTRA = 6
HARD_NEGATIVE_FRAC = 0.10
SEED = 42


def _row_key(row: pd.Series) -> str:
    return sample_id(row["dataset_name"], row["sample_idx"])


def _select_per_dataset(
    pool: pd.DataFrame,
    n_per_dataset: int,
    rng: np.random.Generator,
    exclude: set[str],
    tertile_bounds: dict[str, tuple[float, float]],
) -> list[pd.Series]:
    selected: list[pd.Series] = []
    datasets = sorted(pool["dataset_name"].dropna().unique())

    for ds in datasets:
        sub = pool[pool["dataset_name"] == ds]
        sub = sub[~sub.apply(_row_key, axis=1).isin(exclude)]
        if sub.empty:
            continue

        asym_col = llm_col("steering_asymmetry")
        sub = sub.copy()
        sub["_asym"] = pd.to_numeric(sub.get(asym_col, 0), errors="coerce").fillna(0)
        t33, t66 = tertile_bounds.get(ds, (0.0, 0.0))

        buckets = {"zero": [], "low": [], "mid": [], "high": []}
        for _, row in sub.iterrows():
            buckets[severity_tertile(row["_asym"], t33, t66)].append(row)

        picks: list[pd.Series] = []
        # Prefer spread across tertiles when possible
        order = ["high", "mid", "low", "zero"]
        while len(picks) < n_per_dataset:
            progressed = False
            for bucket in order:
                if buckets[bucket] and len(picks) < n_per_dataset:
                    idx = int(rng.integers(0, len(buckets[bucket])))
                    picks.append(buckets[bucket].pop(idx))
                    progressed = True
            if not progressed:
                break

        if len(picks) < n_per_dataset:
            remaining = sub[~sub.apply(_row_key, axis=1).isin({_row_key(p) for p in picks})]
            extra = remaining.sample(
                n=min(n_per_dataset - len(picks), len(remaining)),
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            picks.extend(row for _, row in extra.iterrows())

        selected.extend(picks[:n_per_dataset])

    return selected


def _tertile_bounds(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds = {}
    col = llm_col("steering_asymmetry")
    for ds in df["dataset_name"].dropna().unique():
        scores = pd.to_numeric(df.loc[df["dataset_name"] == ds, col], errors="coerce").dropna()
        positive = scores[scores > 0]
        if len(positive) < 3:
            bounds[ds] = (0.0, 0.0)
        else:
            bounds[ds] = (
                float(np.percentile(positive, 33)),
                float(np.percentile(positive, 66)),
            )
    return bounds


def _inject_hard_negatives(
    base_rows: list[pd.Series],
    pool: pd.DataFrame,
    rng: np.random.Generator,
    exclude: set[str],
) -> list[pd.Series]:
    n_target = max(1, int(round(len(base_rows) * HARD_NEGATIVE_FRAC)))
    hard_pool = pool[pool.apply(pass2_all_zero, axis=1)]
    hard_pool = hard_pool[~hard_pool.apply(_row_key, axis=1).isin(exclude)]
    if hard_pool.empty or n_target == 0:
        return base_rows

    # Replace random non-gold rows with hard negatives
    replaceable = [i for i, r in enumerate(base_rows) if not r.get("_is_gold", False)]
    n_replace = min(n_target, len(replaceable), len(hard_pool))
    if n_replace == 0:
        return base_rows

    replace_idx = rng.choice(replaceable, size=n_replace, replace=False)
    hard_sample = hard_pool.sample(n=n_replace, random_state=int(rng.integers(0, 2**31 - 1)))

    out = list(base_rows)
    for i, (_, hard_row) in zip(replace_idx, hard_sample.iterrows()):
        out[i] = hard_row
    return out


def _add_high_asymmetry(
    rows: list[pd.Series],
    pool: pd.DataFrame,
    rng: np.random.Generator,
    exclude: set[str],
    n_extra: int,
) -> list[pd.Series]:
    col = llm_col("steering_asymmetry")
    candidates = pool.copy()
    candidates["_asym"] = pd.to_numeric(candidates[col], errors="coerce").fillna(0)
    candidates = candidates[candidates["_asym"] >= 3]
    candidates = candidates[~candidates.apply(_row_key, axis=1).isin(exclude)]
    if candidates.empty or n_extra <= 0:
        return rows

    extra = candidates.nlargest(min(n_extra * 3, len(candidates)), "_asym")
    extra = extra.sample(n=min(n_extra, len(extra)), random_state=int(rng.integers(0, 2**31 - 1)))
    return rows + [row for _, row in extra.iterrows()]


def build_main_set(pass2: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    flagged = pass2[pass2.apply(is_pass1_flagged, axis=1)].copy()
    tertiles = _tertile_bounds(flagged)

    gold = find_gold_rows(flagged)
    gold["_is_gold"] = True
    gold_ids = {_row_key(r) for _, r in gold.iterrows()}

    per_ds = _select_per_dataset(
        flagged,
        N_PER_DATASET,
        rng,
        exclude=gold_ids,
        tertile_bounds=tertiles,
    )

    rows = [row for _, row in gold.iterrows()] + per_ds
    used = {_row_key(r) for r in rows}

    rows = _inject_hard_negatives(rows, flagged, rng, used)
    used = {_row_key(r) for r in rows}

    rows = _add_high_asymmetry(rows, flagged, rng, used, N_HIGH_ASYMMETRY_EXTRA)

    # Trim or pad to target N
    if len(rows) > N_MAIN_TARGET:
        gold_rows = [r for r in rows if r.get("_is_gold", False)]
        other = [r for r in rows if not r.get("_is_gold", False)]
        rng.shuffle(other)
        rows = gold_rows + other[: N_MAIN_TARGET - len(gold_rows)]
    elif len(rows) < N_MAIN_TARGET:
        remaining = flagged[~flagged.apply(_row_key, axis=1).isin({_row_key(r) for r in rows})]
        need = N_MAIN_TARGET - len(rows)
        if len(remaining) >= need:
            extra = remaining.sample(n=need, random_state=seed)
            rows.extend(row for _, row in extra.iterrows())

    out = pd.DataFrame(rows).drop(columns=["_asym", "_is_gold"], errors="ignore")
    out["split"] = "main"
    return out.drop_duplicates(subset=["dataset_name", "sample_idx"])


def build_calibration_set(pass2: pd.DataFrame, main: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    main_ids = {_row_key(r) for _, r in main.iterrows()}
    pool = pass2[pass2.apply(is_pass1_flagged, axis=1)]
    pool = pool[~pool.apply(_row_key, axis=1).isin(main_ids)]

    n = min(N_CALIBRATION, len(pool))
    cal = pool.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1)))
    cal = cal.copy()
    cal["split"] = "calibration"
    return cal


def build_holdout_set(pass1: pd.DataFrame, pass2_ids: set[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    unflagged = pass1[~pass1.apply(is_pass1_flagged, axis=1)].copy()
    rows = []
    for ds in sorted(unflagged["dataset_name"].dropna().unique()):
        sub = unflagged[unflagged["dataset_name"] == ds]
        sub = sub[~sub.apply(_row_key, axis=1).isin(pass2_ids)]
        if sub.empty:
            continue
        pick = sub.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
        rows.append(pick.iloc[0])
    out = pd.DataFrame(rows)
    out["split"] = "holdout"
    return out


def write_manifest(df: pd.DataFrame, path: Path, rater: str, seed: int) -> None:
    order = df.copy()
    order["sample_id"] = order.apply(_row_key, axis=1)
    order = order.sample(frac=1, random_state=seed).reset_index(drop=True)
    order["presentation_order"] = np.arange(1, len(order) + 1)
    order[["sample_id", "presentation_order", "split"]].to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare human validation subsamples")
    parser.add_argument("--pass1", type=Path, default=DEFAULT_PASS1)
    parser.add_argument("--pass2", type=Path, default=DEFAULT_PASS2)
    parser.add_argument("--output-dir", type=Path, default=HUMAN_EVAL_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    pass1 = pd.read_csv(args.pass1, low_memory=False)
    pass2 = pd.read_csv(args.pass2, low_memory=False)

    main_set = build_main_set(pass2, args.seed)
    cal_set = build_calibration_set(pass2, main_set, args.seed)
    all_main_ids = {_row_key(r) for _, r in pd.concat([main_set, cal_set]).iterrows()}
    holdout = build_holdout_set(pass1, all_main_ids, args.seed)

    combined = pd.concat([cal_set, main_set, holdout], ignore_index=True)
    combined = attach_llm_refs(combined)
    combined["sample_id"] = combined.apply(_row_key, axis=1)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("calibration", "main", "holdout"):
        part = combined[combined["split"] == split_name]
        part.to_csv(out_dir / f"subsample_{split_name}.csv", index=False)

    combined.to_csv(out_dir / "subsample_all.csv", index=False)

    meta = {
        "seed": args.seed,
        "n_calibration": len(cal_set),
        "n_main": len(main_set),
        "n_holdout": len(holdout),
        "gold_cases": GOLD_CASE_QUERIES,
        "primary_dimensions": PRIMARY_DIMENSIONS,
    }
    (out_dir / "subsample_manifest.json").write_text(json.dumps(meta, indent=2))

    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    for split_name in ("calibration", "main", "holdout"):
        part = combined[combined["split"] == split_name]
        for rater, offset in (("a", 100), ("b", 200)):
            write_manifest(
                part,
                manifest_dir / f"{split_name}_order_rater_{rater}.csv",
                rater,
                args.seed + offset + hash(split_name) % 50,
            )

    print(f"Wrote calibration: {len(cal_set)}, main: {len(main_set)}, holdout: {len(holdout)}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
