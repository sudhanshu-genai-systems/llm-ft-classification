"""
evaluate.py
-----------
Run classification evaluation and aggregate metrics across splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from transformers import Trainer, TrainingArguments

try:
    from finetune.metrics import build_compute_metrics
    from finetune.settings.app import EvaluationSettings, TrainingSettings
except ImportError:
    from metrics import build_compute_metrics
    from settings.app import EvaluationSettings, TrainingSettings


def _tokenized_subset(tokenized: Any, max_samples: int | None):
    if max_samples is None or max_samples >= len(tokenized):
        return tokenized
    return tokenized.select(range(max_samples))


def evaluate_splits(
    model: Any,
    tokenized_datasets: dict[str, Any],
    training: TrainingSettings,
    evaluation: EvaluationSettings,
    output_dir: Path,
    phase: str,
) -> dict[str, dict[str, float]]:
    """
    Evaluate ``model`` on configured splits.

    phase: short label stored in results (e.g. ``baseline`` or ``after_lora``).
    """
    args = TrainingArguments(
        output_dir=str(output_dir / f"eval_{phase}"),
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        bf16=training.bf16,
        fp16=training.fp16,
        report_to=[],
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        compute_metrics=build_compute_metrics(),
    )

    results: dict[str, dict[str, float]] = {}
    for split in evaluation.splits:
        if split not in tokenized_datasets:
            raise KeyError(f"Split {split!r} not found in tokenized datasets.")
        ds = _tokenized_subset(
            tokenized_datasets[split],
            evaluation.max_eval_samples,
        )
        metrics = trainer.evaluate(eval_dataset=ds, metric_key_prefix=split)
        cleaned = {
            k.replace(f"{split}_", ""): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating))
        }
        results[split] = cleaned
    return results


def compare_evaluations(
    baseline: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
    primary_metric: str = "f1_macro",
) -> dict[str, dict[str, float]]:
    delta: dict[str, dict[str, float]] = {}
    for split in baseline:
        if split not in after:
            continue
        delta[split] = {}
        keys = set(baseline[split]) | set(after[split])
        for key in keys:
            if key in baseline[split] and key in after[split]:
                delta[split][key] = after[split][key] - baseline[split][key]
        if primary_metric in delta[split]:
            delta[split]["primary_metric"] = primary_metric
    return delta
