"""
data_analysis.py
----------------
Exploratory Data Analysis for BANKING77.

Run as a script:
    python data_analysis.py --out reports/eda

Or import in a notebook (repo root on ``sys.path``):

    from finetune.data_analysis import run_eda
    summary = run_eda(train_df, test_df, out_dir="reports/eda", show=True)

Produces:
  - console summary (sizes, class balance, text-length stats, data-quality checks)
  - class_distribution.png, text_length_distribution.png, tokens_per_class.png
  - eda_summary.csv (per-intent stats you can sort/filter in the notebook)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")  # safe default for scripts; notebooks can override
import matplotlib.pyplot as plt
import pandas as pd

try:
    from .load_data import load_banking77
except ImportError:
    from load_data import load_banking77


# ----------------------------- core stats ---------------------------------

def per_class_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Per-intent counts and text-length stats (words), train vs test."""

    def agg(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        g = df.assign(n_words=df["text"].str.split().str.len()).groupby("category")
        return cast(
            pd.DataFrame,
            g.agg(
                **{
                    f"{prefix}_count": ("text", "size"),
                    f"{prefix}_avg_words": ("n_words", "mean"),
                    f"{prefix}_max_words": ("n_words", "max"),
                }
            ),
        )

    stats = agg(train_df, "train").join(agg(test_df, "test"), how="outer")
    stats["train_share_%"] = 100 * stats["train_count"] / stats["train_count"].sum()
    return stats.sort_values("train_count", ascending=False).round(2)


def quality_checks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, int]:
    """Duplicates, empty texts, and train/test leakage."""
    train_texts = train_df["text"].str.strip().str.lower()
    test_texts = test_df["text"].str.strip().str.lower()

    dup_train = int(train_texts.duplicated().sum())
    dup_test = int(test_texts.duplicated().sum())
    leakage = int(test_texts.isin(set(train_texts)).sum())

    # same text labeled with different intents (annotation conflicts)
    conflicts = train_df.assign(t=train_texts).groupby("t")["category"].nunique()
    conflict_count = int((conflicts > 1).sum())

    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "n_intents_train": cast(int, train_df["category"].nunique(dropna=False)),
        "n_intents_test": cast(int, test_df["category"].nunique(dropna=False)),
        "duplicate_texts_train": dup_train,
        "duplicate_texts_test": dup_test,
        "train_test_overlap_texts": leakage,
        "label_conflicts_train": conflict_count,
        "empty_texts_train": int((train_texts.str.len() == 0).sum()),
        "empty_texts_test": int((test_texts.str.len() == 0).sum()),
    }


# ------------------------------- plots -------------------------------------

def plot_class_distribution(train_df: pd.DataFrame, out: Path) -> None:
    counts = train_df["category"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 16))
    counts.sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_title("BANKING77 - train examples per intent")
    ax.set_xlabel("count")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_text_lengths(train_df: pd.DataFrame, test_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    splits: list[tuple[str, pd.DataFrame]] = [("train", train_df), ("test", test_df)]
    for ax, (name, df) in zip(axes, splits, strict=True):
        lengths = df["text"].str.split().str.len()
        ax.hist(lengths, bins=40, color="#55A868")
        ax.set_title(
            f"{name}: words per query (p95={lengths.quantile(0.95):.0f}, max={lengths.max()})"
        )
        ax.set_xlabel("words")
    fig.tight_layout()
    fig.savefig(out / "text_length_distribution.png", dpi=150)
    plt.close(fig)


def plot_length_by_class(train_df: pd.DataFrame, out: Path, top_n: int = 20) -> None:
    lengths = train_df.assign(n_words=train_df["text"].str.split().str.len())
    means = cast(pd.Series, lengths.groupby("category")["n_words"].mean())
    top = means.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    top.sort_values(ascending=True).plot.barh(ax=ax, color="#C44E52")
    ax.set_title(f"Top {top_n} intents by average query length (train)")
    ax.set_xlabel("avg words")
    fig.tight_layout()
    fig.savefig(out / "tokens_per_class.png", dpi=150)
    plt.close(fig)


# ------------------------------- driver -------------------------------------

def run_eda(
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    out_dir: str | Path = "reports/eda",
    show: bool = False,
) -> pd.DataFrame:
    """Run full EDA. Returns the per-intent stats table."""
    if train_df is None or test_df is None:
        loaded_train, loaded_test = load_banking77()
        if train_df is None:
            train_df = loaded_train
        if test_df is None:
            test_df = loaded_test

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    checks = quality_checks(train_df, test_df)
    print("=== Data quality / shape ===")
    for k, v in checks.items():
        print(f"  {k:>28}: {v:,}")

    stats = per_class_stats(train_df, test_df)
    stats.to_csv(out / "eda_summary.csv")

    imbalance = stats["train_count"].max() / stats["train_count"].min()
    print("\n=== Class balance (train) ===")
    print(f"  largest intent : {stats['train_count'].idxmax()} ({stats['train_count'].max():.0f})")
    print(f"  smallest intent: {stats['train_count'].idxmin()} ({stats['train_count'].min():.0f})")
    print(f"  imbalance ratio: {imbalance:.1f}x")

    lengths = train_df["text"].str.split().str.len()
    print("\n=== Text length (train, words) ===")
    print(
        f"  mean={lengths.mean():.1f}  median={lengths.median():.0f}  "
        f"p95={lengths.quantile(0.95):.0f}  max={lengths.max()}"
    )
    print("  -> use p95/max to pick tokenizer max_length for fine-tuning")

    plot_class_distribution(train_df, out)
    plot_text_lengths(train_df, test_df, out)
    plot_length_by_class(train_df, out)
    print(f"\nPlots + eda_summary.csv written to: {out.resolve()}")

    if show:  # in a notebook, also display inline
        try:
            from IPython.display import Image, display
        except ImportError:
            for img in [
                "class_distribution.png",
                "text_length_distribution.png",
                "tokens_per_class.png",
            ]:
                print(f"(open {out / img})")
        else:
            for img in [
                "class_distribution.png",
                "text_length_distribution.png",
                "tokens_per_class.png",
            ]:
                display(Image(filename=str(out / img)))

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA for BANKING77")
    parser.add_argument("--out", default="reports/eda", help="output directory for plots/CSV")
    args = parser.parse_args()
    run_eda(out_dir=args.out)
