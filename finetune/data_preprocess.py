"""
data_preprocess.py
------------------
Preprocessing pipeline for BANKING77 -> model-ready splits.

Steps (each togglable):
  1. Load train/test from the canonical CSVs (same source as the old HF script).
  2. Encode string categories to the canonical integer ids (0-76).
  3. Light text cleaning (whitespace normalization; intent text is kept as-is
     otherwise - casing and punctuation carry signal for intent detection).
  4. Deduplicate exact-duplicate (text, label) pairs in train.
  5. Stratified train/validation split (test set is left untouched as the
     final benchmark, same protocol as the original papers).
  6. Save parquet files + label mapping JSON to an output dir.

Run:
    python data_preprocess.py --val-size 0.1 --seed 42 --out data/processed

Notebook (add repo root to path, e.g. ``sys.path.insert(0, "..")`` from ``notebooks/``):

    from finetune.data_preprocess import preprocess
    splits = preprocess(val_size=0.1, seed=42, out_dir="data/processed")
    splits["train"].head()

The saved parquet files load directly into HF ``datasets`` for fine-tuning:

    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={
        "train": "data/processed/train.parquet",
        "validation": "data/processed/validation.parquet",
        "test": "data/processed/test.parquet",
    })
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Literal, TypedDict, cast

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .load_data import LABELS, id2label, label2id, load_banking77
except ImportError:
    from load_data import LABELS, id2label, label2id, load_banking77

_WS = re.compile(r"\s+")


class Banking77Splits(TypedDict):
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def clean_text(s: pd.Series) -> pd.Series:
    """Minimal, safe cleaning: strip + collapse internal whitespace.

    Deliberately does NOT lowercase or strip punctuation: subword tokenizers
    handle both, and question marks / phrasing are useful intent signals.
    """

    def _collapse_ws(text: str) -> str:
        return _WS.sub(" ", text)

    return s.astype(str).str.strip().map(_collapse_ws)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    deduped = df.drop_duplicates(subset=["text", "label"], keep="first").reset_index(drop=True)
    return deduped, before - len(deduped)


def make_validation_split(
    train_df: pd.DataFrame, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, va = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            train_df,
            test_size=val_size,
            random_state=seed,
            stratify=train_df["label"],
        ),
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def preprocess(
    val_size: float = 0.1,
    seed: int = 42,
    out_dir: str | Path | None = "data/processed",
    dedupe: bool = True,
) -> Banking77Splits:
    """Full pipeline. Returns train, validation, and test DataFrames."""
    train_df, test_df = load_banking77()

    for df in (train_df, test_df):
        text_col = cast(pd.Series, df["text"])
        df["text"] = clean_text(text_col)

    if dedupe:
        train_df, removed = deduplicate(train_df)
        print(f"deduplicated train: removed {removed} exact (text,label) duplicates")

    tr, va = make_validation_split(train_df, val_size, seed)

    splits: Banking77Splits = {"train": tr, "validation": va, "test": test_df}
    split_names: tuple[Literal["train", "validation", "test"], ...] = (
        "train",
        "validation",
        "test",
    )
    for name in split_names:
        df = splits[name]
        counts = df["label"].value_counts()
        print(
            f"{name:>10}: {len(df):>6,} rows | {df['label'].nunique()} intents | "
            f"min/class={counts.min()}, max/class={counts.max()}"
        )

    # sanity: every intent present in every split
    for name in split_names:
        df = splits[name]
        missing = set(range(len(LABELS))) - set(df["label"].unique())
        if missing:
            raise RuntimeError(
                f"{name} split is missing intents: {sorted(missing)} "
                f"- lower val_size or reduce dedupe"
            )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            for name in split_names:
                df = splits[name]
                df[["text", "category", "label"]].to_parquet(out / f"{name}.parquet", index=False)
        except ImportError:
            print(
                "pyarrow/fastparquet not installed - saving CSV instead "
                "(pip install pyarrow for parquet)"
            )
            for name in split_names:
                df = splits[name]
                df[["text", "category", "label"]].to_csv(out / f"{name}.csv", index=False)
        (out / "label_mapping.json").write_text(
            json.dumps({"label2id": label2id, "id2label": id2label}, indent=2)
        )
        print(f"wrote parquet splits + label_mapping.json to: {out.resolve()}")

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BANKING77")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args()
    preprocess(
        val_size=args.val_size,
        seed=args.seed,
        out_dir=args.out,
        dedupe=not args.no_dedupe,
    )
