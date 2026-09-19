"""
classification_data.py
----------------------
Load processed BANKING77 splits for sequence classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

try:
    from finetune.load_data import LABELS, id2label, label2id
except ImportError:
    from load_data import LABELS, id2label, label2id


def _split_files(data_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in ("train", "validation", "test"):
        parquet = data_dir / f"{name}.parquet"
        csv = data_dir / f"{name}.csv"
        if parquet.exists():
            paths[name] = parquet
        elif csv.exists():
            paths[name] = csv
        else:
            raise FileNotFoundError(
                f"Missing {name} split under {data_dir} (expected .parquet or .csv). "
                "Run finetune/data_preprocess.py --out notebooks/data/processed first."
            )
    return paths


def load_label_maps(data_dir: Path, mapping_file: str) -> tuple[dict[str, int], dict[int, str]]:
    mapping_path = data_dir / mapping_file
    if mapping_path.exists():
        raw = json.loads(mapping_path.read_text())
        l2i = {str(k): int(v) for k, v in raw["label2id"].items()}
        i2l = {int(k): str(v) for k, v in raw["id2label"].items()}
        return l2i, i2l
    return label2id, id2label


def load_classification_datasets(
    data_dir: str | Path,
    label_mapping_file: str = "label_mapping.json",
    max_train_samples: int | None = None,
) -> tuple[DatasetDict, dict[str, int], dict[int, str], int]:
    """
    Load train / validation / test splits from processed parquet or CSV.

    Returns (dataset_dict, label2id, id2label, num_labels).
    """
    data_path = Path(data_dir).resolve()
    files = _split_files(data_path)
    l2i, i2l = load_label_maps(data_path, label_mapping_file)
    num_labels = len(l2i)

    def _load(path: Path) -> Dataset:
        if path.suffix == ".parquet":
            return Dataset.from_parquet(str(path))
        return Dataset.from_csv(str(path))

    ds = DatasetDict({name: _load(path) for name, path in files.items()})

    for split in ds:
        if "label" not in ds[split].column_names:
            raise ValueError(f"Split {split} is missing a 'label' column.")
        if "text" not in ds[split].column_names:
            raise ValueError(f"Split {split} is missing a 'text' column.")

    if max_train_samples is not None:
        n = min(max_train_samples, len(ds["train"]))
        ds["train"] = ds["train"].select(range(n))

    # Ensure canonical label count matches BANKING77 when using default maps.
    if num_labels != len(LABELS):
        raise ValueError(f"Expected {len(LABELS)} labels, found {num_labels} in mapping.")

    return ds, l2i, i2l, num_labels


def subset_for_eval(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))
