"""Pass-1 vs Pass-2 comparison tables and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .constants import PASS1_CATEGORIES, PASS2_CATEGORY_EFFECTS, STEERABILITY_ORDER
from .utils import add_pass1_derived, effect_columns, is_flagged, severity_matrix


def build_funnel(pass1: pd.DataFrame, pass2: pd.DataFrame) -> pd.DataFrame:
    p1 = add_pass1_derived(pass1)
    p2 = add_pass1_derived(pass2)
    n_total = len(p1)
    n_flagged = int(p1["any_category_flagged"].sum())

    p2_flagged = p2[p2["any_category_flagged"]].copy()
    n_pass2 = len(p2_flagged)

    sev = severity_matrix(p2_flagged)
    n_any_nonzero = int((sev.max(axis=1) > 0).sum())
    n_all_zero = int((sev.max(axis=1) == 0).sum())

    rows = [
        {"stage": "total_corpus", "metric": "n", "value": n_total},
        {"stage": "pass1_any_category_flagged", "metric": "n", "value": n_flagged},
        {"stage": "pass1_any_category_flagged", "metric": "pct_of_corpus", "value": round(n_flagged / n_total * 100, 2)},
        {"stage": "pass2_evaluated", "metric": "n", "value": n_pass2},
        {"stage": "pass2_evaluated", "metric": "pct_of_corpus", "value": round(n_pass2 / n_total * 100, 2)},
        {"stage": "pass2_any_severity_nonzero", "metric": "n", "value": n_any_nonzero},
        {"stage": "pass2_any_severity_nonzero", "metric": "pct_of_pass2", "value": round(n_any_nonzero / n_pass2 * 100, 2) if n_pass2 else 0},
        {"stage": "pass2_all_severity_zero", "metric": "n", "value": n_all_zero},
        {"stage": "pass2_all_severity_zero", "metric": "pct_of_pass2", "value": round(n_all_zero / n_pass2 * 100, 2) if n_pass2 else 0},
    ]

    if "judge_steerability" in p2_flagged.columns:
        for steer in STEERABILITY_ORDER:
            sub = p2_flagged[p2_flagged["judge_steerability"].astype(str).str.upper() == steer]
            if sub.empty:
                continue
            sub_sev = severity_matrix(sub)
            rows.append({"stage": f"steerability_{steer.lower()}", "metric": "n", "value": len(sub)})
            rows.append({
                "stage": f"steerability_{steer.lower()}",
                "metric": "pct_any_pass2_nonzero",
                "value": round((sub_sev.max(axis=1) > 0).mean() * 100, 2),
            })
            rows.append({
                "stage": f"steerability_{steer.lower()}",
                "metric": "mean_max_severity",
                "value": round(sub_sev.max(axis=1).mean(), 3),
            })

    return pd.DataFrame(rows)


def build_concordance(pass2: pd.DataFrame, label: str = "full") -> pd.DataFrame:
    p2 = add_pass1_derived(pass2)
    p2 = p2[p2["any_category_flagged"]]
    rows = []

    for category, dimensions in PASS2_CATEGORY_EFFECTS.items():
        flag_col = f"flag_{category}"
        flagged = p2[p2[flag_col]]
        n_flagged = len(flagged)
        if n_flagged == 0:
            continue

        for dim in dimensions:
            col = f"judge_effect_{dim}"
            if col not in flagged.columns:
                continue
            scores = pd.to_numeric(flagged[col], errors="coerce").fillna(0)
            rows.append({
                "subset": label,
                "pass1_category": category,
                "pass2_dimension": dim,
                "n_pass1_flagged": n_flagged,
                "pct_nonzero": round((scores > 0).mean() * 100, 2),
                "pct_severity_ge3": round((scores >= 3).mean() * 100, 2),
                "mean_severity": round(scores.mean(), 4),
            })

    return pd.DataFrame(rows)


def build_orphan_diagnostic(pass2: pd.DataFrame, label: str = "full") -> pd.DataFrame:
    """Concordant vs orphan Pass-2 prevalence within the Pass-1-flagged subset.

    Concordant: parent Pass-1 category flagged. Orphan: parent not flagged but sample
    reached Pass 2 via another category flag.
    """
    p2 = add_pass1_derived(pass2)
    p2 = p2[p2["any_category_flagged"]]
    rows = []

    for category, dimensions in PASS2_CATEGORY_EFFECTS.items():
        flag_col = f"flag_{category}"
        parent_flagged = p2[flag_col]
        n_concordant = int(parent_flagged.sum())
        n_orphan = int((~parent_flagged).sum())
        if n_concordant == 0 and n_orphan == 0:
            continue

        for dim in dimensions:
            col = f"judge_effect_{dim}"
            if col not in p2.columns:
                continue
            scores = pd.to_numeric(p2[col], errors="coerce").fillna(0)
            concordant = scores[parent_flagged]
            orphan = scores[~parent_flagged]
            pct_concordant = round((concordant > 0).mean() * 100, 2) if n_concordant else 0.0
            pct_orphan = round((orphan > 0).mean() * 100, 2) if n_orphan else 0.0
            rows.append({
                "subset": label,
                "pass1_category": category,
                "pass2_dimension": dim,
                "n_concordant": n_concordant,
                "n_orphan": n_orphan,
                "pct_nonzero_concordant": pct_concordant,
                "pct_nonzero_orphan": pct_orphan,
                "concordance_gap_pp": round(pct_concordant - pct_orphan, 2),
                "pct_severity_ge3_concordant": round((concordant >= 3).mean() * 100, 2) if n_concordant else 0.0,
                "pct_severity_ge3_orphan": round((orphan >= 3).mean() * 100, 2) if n_orphan else 0.0,
            })

    return pd.DataFrame(rows)


def build_steerability_vs_severity(pass2: pd.DataFrame, label: str = "full") -> pd.DataFrame:
    p2 = add_pass1_derived(pass2)
    p2 = p2[p2["any_category_flagged"]]
    if "judge_steerability" not in p2.columns:
        return pd.DataFrame()

    rows = []
    for steer in sorted(p2["judge_steerability"].dropna().astype(str).str.upper().unique()):
        sub = p2[p2["judge_steerability"].astype(str).str.upper() == steer]
        sev = severity_matrix(sub)
        for col in effect_columns(sub):
            dim = col.replace("judge_effect_", "")
            scores = sev[col]
            rows.append({
                "subset": label,
                "steerability": steer,
                "dimension": dim,
                "n": len(sub),
                "mean_severity": round(scores.mean(), 4),
                "max_severity": int(scores.max()) if len(scores) else 0,
                "pct_nonzero": round((scores > 0).mean() * 100, 2),
            })

    return pd.DataFrame(rows)


def build_cooccurrence(pass2: pd.DataFrame, label: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
    p2 = add_pass1_derived(pass2)
    p2 = p2[p2["any_category_flagged"]]
    sev = severity_matrix(p2)
    p2 = p2.copy()
    p2["max_severity"] = sev.max(axis=1)
    p2["mean_severity"] = sev.mean(axis=1)

    rows = []
    for n_cats in sorted(p2["n_categories_flagged"].unique()):
        sub = p2[p2["n_categories_flagged"] == n_cats]
        if sub.empty:
            continue
        rows.append({
            "subset": label,
            "n_categories_flagged": int(n_cats),
            "n_samples": len(sub),
            "pct_of_pass2": round(len(sub) / len(p2) * 100, 2),
            "pct_any_pass2_nonzero": round((sub["max_severity"] > 0).mean() * 100, 2),
            "mean_max_severity": round(sub["max_severity"].mean(), 3),
            "mean_avg_severity": round(sub["mean_severity"].mean(), 3),
        })

    pair_rows = []
    for i, c1 in enumerate(PASS1_CATEGORIES):
        for c2 in PASS1_CATEGORIES[i + 1:]:
            both = p2[p2[f"flag_{c1}"] & p2[f"flag_{c2}"]]
            if both.empty:
                continue
            pair_rows.append({
                "subset": label,
                "category_a": c1,
                "category_b": c2,
                "n_co_flagged": len(both),
                "mean_max_severity": round(both["max_severity"].mean(), 3),
            })

    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def plot_concordance_heatmap(concordance: pd.DataFrame, path: Path) -> None:
    full = concordance[concordance["subset"] == "full"].copy()
    if full.empty:
        return

    pivot = full.pivot(index="pass1_category", columns="pass2_dimension", values="pct_nonzero")
    dim_order = []
    for cat in PASS1_CATEGORIES:
        for dim in PASS2_CATEGORY_EFFECTS.get(cat, []):
            if dim in pivot.columns and dim not in dim_order:
                dim_order.append(dim)
    pivot = pivot.reindex(index=PASS1_CATEGORIES, columns=dim_order)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "% non-zero Pass-2 severity"},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Pass-1 category flag → Pass-2 dimension prevalence (% non-zero, among flagged category)")
    ax.set_xlabel("Pass-2 dimension")
    ax.set_ylabel("Pass-1 category")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_steerability_boxplot(pass2: pd.DataFrame, path: Path) -> None:
    p2 = add_pass1_derived(pass2)
    p2 = p2[p2["any_category_flagged"]]
    col = "judge_effect_steering_asymmetry"
    if col not in p2.columns or "judge_steerability" not in p2.columns:
        return

    plot_df = p2[["judge_steerability", col]].copy()
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df["steerability"] = plot_df["judge_steerability"].astype(str).str.upper()
    order = [s for s in STEERABILITY_ORDER if s in plot_df["steerability"].unique()]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=plot_df, x="steerability", y=col, order=order, ax=ax, hue="steerability", legend=False)
    ax.set_xlabel("Pass-1 steerability")
    ax.set_ylabel("Pass-2 steering asymmetry severity")
    ax.set_title("Steering asymmetry severity by Pass-1 steerability label")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(
    funnel: pd.DataFrame,
    concordance: pd.DataFrame,
    cooccur: pd.DataFrame,
    orphans: pd.DataFrame,
    path: Path,
) -> None:
    def get(stage: str, metric: str):
        row = funnel[(funnel["stage"] == stage) & (funnel["metric"] == metric)]
        return row["value"].iloc[0] if not row.empty else None

    full_conc = concordance[concordance["subset"] == "full"]
    full_orphans = orphans[orphans["subset"] == "full"]
    top = full_conc.sort_values("pct_nonzero", ascending=False).head(8)
    low = full_conc.sort_values("pct_nonzero").head(5)
    top_gaps = full_orphans.sort_values("concordance_gap_pp", ascending=False).head(8)

    lines = [
        "# Pass 1 vs Pass 2 Comparison Summary",
        "",
        "## Screening funnel (full run)",
        "",
        f"- Total corpus: **{int(get('total_corpus', 'n')):,}**",
        f"- Pass-1 any category flagged: **{int(get('pass1_any_category_flagged', 'n')):,}** ({get('pass1_any_category_flagged', 'pct_of_corpus')}%)",
        f"- Pass-2 evaluated: **{int(get('pass2_evaluated', 'n')):,}** ({get('pass2_evaluated', 'pct_of_corpus')}% of corpus)",
        f"- Pass-2 any severity > 0: **{int(get('pass2_any_severity_nonzero', 'n')):,}** ({get('pass2_any_severity_nonzero', 'pct_of_pass2')}% of Pass-2 rows)",
        f"- Pass-2 all severities = 0: **{int(get('pass2_all_severity_zero', 'n')):,}** ({get('pass2_all_severity_zero', 'pct_of_pass2')}% of Pass-2 rows)",
        "",
        "## Highest category→dimension concordance",
        "",
        top.to_markdown(index=False),
        "",
        "## Lowest concordance (among nested pairs)",
        "",
        low.to_markdown(index=False),
        "",
        "## Largest concordant vs orphan gaps (Pass-1 screening precision)",
        "",
        top_gaps.to_markdown(index=False),
        "",
        "## Category co-occurrence vs Pass-2 severity",
        "",
        cooccur[cooccur["subset"] == "full"].to_markdown(index=False),
        "",
        "## Caveats",
        "",
        "- Pass 2 runs only on Pass-1-flagged samples; corpus-level Pass-1 false negatives are unobservable.",
        "- Orphan rates quantify cross-category leakage among Pass-2-evaluated rows (parent category not flagged).",
        "- Pass 1 category flags are injected into the Pass 2 prompt (`flagged_categories`).",
    ]
    path.write_text("\n".join(lines))


def run_comparison(
    pass1_path: Path,
    pass2_path: Path,
    output_dir: Path,
    iaa_pass1: Path | None = None,
    iaa_pass2: Path | None = None,
) -> dict:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    notes_dir = output_dir / "notes"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Pass 1: {pass1_path}")
    pass1 = pd.read_csv(pass1_path, low_memory=False)
    print(f"Loading Pass 2: {pass2_path}")
    pass2 = pd.read_csv(pass2_path, low_memory=False)

    funnel = build_funnel(pass1, pass2)
    funnel["subset"] = "full"
    funnel.to_csv(tables_dir / "pass1_pass2_funnel.csv", index=False)

    concordance = build_concordance(pass2, label="full")
    orphans = build_orphan_diagnostic(pass2, label="full")
    steerability = build_steerability_vs_severity(pass2, label="full")
    cooccur, cooccur_pairs = build_cooccurrence(pass2, label="full")

    if iaa_pass1 and iaa_pass2 and iaa_pass1.exists() and iaa_pass2.exists():
        print("Running IAA subsample comparison...")
        iaa_p2 = pd.read_csv(iaa_pass2, low_memory=False)
        iaa_funnel = build_funnel(pd.read_csv(iaa_pass1, low_memory=False), iaa_p2)
        iaa_funnel["subset"] = "iaa"
        funnel = pd.concat([funnel, iaa_funnel], ignore_index=True)
        funnel.to_csv(tables_dir / "pass1_pass2_funnel.csv", index=False)

        iaa_conc = build_concordance(iaa_p2, label="iaa")
        concordance = pd.concat([concordance, iaa_conc], ignore_index=True)

        iaa_orphans = build_orphan_diagnostic(iaa_p2, label="iaa")
        orphans = pd.concat([orphans, iaa_orphans], ignore_index=True)

        iaa_steer = build_steerability_vs_severity(iaa_p2, label="iaa")
        steerability = pd.concat([steerability, iaa_steer], ignore_index=True)

        iaa_co, iaa_pairs = build_cooccurrence(iaa_p2, label="iaa")
        cooccur = pd.concat([cooccur, iaa_co], ignore_index=True)
        cooccur_pairs = pd.concat([cooccur_pairs, iaa_pairs], ignore_index=True)

    concordance.to_csv(tables_dir / "pass1_pass2_concordance.csv", index=False)
    orphans.to_csv(tables_dir / "pass1_pass2_orphans.csv", index=False)
    steerability.to_csv(tables_dir / "pass1_steerability_vs_pass2_severity.csv", index=False)
    cooccur.to_csv(tables_dir / "pass1_category_cooccurrence.csv", index=False)
    cooccur_pairs.to_csv(tables_dir / "pass1_category_cooccurrence_pairs.csv", index=False)

    plot_concordance_heatmap(concordance, figures_dir / "pass1_pass2_concordance_heatmap.png")
    plot_steerability_boxplot(pass2, figures_dir / "pass1_steerability_vs_asymmetry_boxplot.png")

    write_summary_md(
        funnel[funnel["subset"] == "full"],
        concordance,
        cooccur,
        orphans,
        notes_dir / "pass1_pass2_comparison.md",
    )

    print(f"Wrote Pass-1/Pass-2 tables to {tables_dir}")
    print(f"Wrote Pass-1/Pass-2 figures to {figures_dir}")

    return {
        "n_total": int(funnel[(funnel["stage"] == "total_corpus") & (funnel["metric"] == "n")]["value"].iloc[0]),
        "n_pass2": int(funnel[(funnel["stage"] == "pass2_evaluated") & (funnel["metric"] == "n")]["value"].iloc[0]),
    }
