"""Orchestrate thesis table and figure regeneration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import shutil

from .config import THESIS_ROOT, resolve_paths
from .examples import extract_qualitative_examples
from .figures import generate_figures
from .pass1_pass2 import run_comparison
from .per_alpha import run_per_alpha_metrics
from .perplexity import run_perplexity_analysis
from .tables import export_iaa_metrics, generate_tables


def sync_thesis_figures(results_dir: Path) -> None:
    """Copy generated PNGs into thesis/figures/ for Overleaf (thesis-only uploads)."""
    src = results_dir / "figures"
    dst = THESIS_ROOT / "thesis" / "figures"
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for png in sorted(src.glob("*.png")):
        shutil.copy2(png, dst / png.name)
    print(f"Synced {len(list(dst.glob('*.png')))} figure(s) to {dst}")


def regenerate(
    data_root: Path | None = None,
    output_dir: Path | None = None,
    *,
    skip_tables: bool = False,
    skip_figures: bool = False,
    skip_pass1_pass2: bool = False,
    skip_per_alpha: bool = False,
    skip_perplexity: bool = False,
    skip_examples: bool = False,
    skip_iaa: bool = False,
    include_iaa_subsample: bool = True,
    max_rows: int | None = None,
) -> None:
    paths = resolve_paths(data_root=data_root, output_dir=output_dir)

    for label, path in [
        ("Pass-1 CSV", paths.pass1_csv),
        ("Pass-2 CSV", paths.pass2_csv),
        ("Significance CSV", paths.significance_csv),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    print(f"Data root: {paths.data_root}")
    print(f"Output dir: {paths.output_dir}")

    pass1_df = pd.read_csv(paths.pass1_csv, low_memory=False)
    pass2_df = pd.read_csv(paths.pass2_csv, low_memory=False)
    sig_df = pd.read_csv(paths.significance_csv)

    if not skip_tables:
        generate_tables(
            pass1_df,
            pass2_df,
            sig_df,
            paths.output_dir,
            iaa_agreement_path=paths.iaa_agreement_csv,
            significance_csv=paths.significance_csv,
        )

    if not skip_figures:
        generate_figures(sig_df, pass1_df, pass2_df, paths.output_dir)

    if not skip_examples:
        extract_qualitative_examples(pass2_df, paths.output_dir)

    if not skip_iaa:
        export_iaa_metrics(paths.iaa_agreement_csv, paths.output_dir)

    if not skip_pass1_pass2:
        print("Running Pass-1 vs Pass-2 comparison...")
        iaa_p1 = paths.iaa_pass1_csv if include_iaa_subsample else None
        iaa_p2 = paths.iaa_pass2_csv if include_iaa_subsample else None
        run_comparison(
            paths.pass1_csv,
            paths.pass2_csv,
            paths.output_dir,
            iaa_pass1=iaa_p1,
            iaa_pass2=iaa_p2,
        )

    if not skip_per_alpha:
        run_per_alpha_metrics(paths.pass2_csv, paths.output_dir, max_rows=max_rows)

    if not skip_perplexity:
        run_perplexity_analysis(
            paths.generations_dir,
            paths.pass2_csv,
            paths.output_dir,
            max_rows=max_rows,
        )

    if not (
        skip_figures
        and skip_pass1_pass2
        and skip_per_alpha
        and skip_perplexity
    ):
        sync_thesis_figures(paths.output_dir)

    print("Done!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate thesis results tables and figures.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory for data/ inputs (default: data/ in this repo)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thesis results output directory (default: results/)",
    )
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-pass1-pass2", action="store_true")
    parser.add_argument("--skip-per-alpha", action="store_true")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--skip-examples", action="store_true")
    parser.add_argument("--skip-iaa", action="store_true")
    parser.add_argument("--no-iaa-subsample", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit Pass-2 rows for testing")
    args = parser.parse_args()

    regenerate(
        data_root=args.data_root,
        output_dir=args.output_dir,
        skip_tables=args.skip_tables,
        skip_figures=args.skip_figures,
        skip_pass1_pass2=args.skip_pass1_pass2,
        skip_per_alpha=args.skip_per_alpha,
        skip_perplexity=args.skip_perplexity,
        skip_examples=args.skip_examples,
        skip_iaa=args.skip_iaa,
        include_iaa_subsample=not args.no_iaa_subsample,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
