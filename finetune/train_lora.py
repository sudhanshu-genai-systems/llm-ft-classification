"""
train_lora.py
-------------
LoRA fine-tuning pipeline for BANKING77 intent classification on Llama 3B.

Evaluates the model before training (baseline) and after LoRA fine-tuning, then
writes a JSON comparison under the configured output directory.

Notebook (from ``notebooks/``, repo root on ``sys.path``):

    from finetune.train_lora import run_lora_finetune
    from finetune.settings import FinetuneConfig
    run_lora_finetune(FinetuneConfig.load())

CLI:

    python finetune/run_lora_finetune.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

from finetune.classification_data import load_classification_datasets
from finetune.evaluate import compare_evaluations, evaluate_splits
from finetune.metrics import build_compute_metrics
from finetune.model_factory import (
    apply_lora,
    create_base_sequence_classifier,
    create_tokenizer,
    prepare_model_for_training,
)
from finetune.settings import FinetuneConfig


def _ensure_dirs(config: FinetuneConfig) -> None:
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    config.paths.cache_dir.mkdir(parents=True, exist_ok=True)


def _tokenize_datasets(
    raw_datasets: Any,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    def preprocess(examples: dict[str, list]) -> dict[str, Any]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    tokenized: dict[str, Any] = {}
    for split, ds in raw_datasets.items():
        cols = ds.column_names
        remove = [c for c in cols if c not in ("text", "label")]
        tokenized[split] = ds.map(
            preprocess,
            batched=True,
            remove_columns=remove,
            desc=f"Tokenizing {split}",
        )
    return tokenized


def run_lora_finetune(config: FinetuneConfig | None = None) -> dict[str, Any]:
    """Full pipeline: baseline eval → LoRA train → post-train eval → save artifacts."""
    config = config or FinetuneConfig.load()
    _ensure_dirs(config)
    set_seed(config.training.seed)

    raw_datasets, label2id_map, id2label_map, num_labels = load_classification_datasets(
        config.paths.data_dir,
        label_mapping_file=config.paths.label_mapping_file,
        max_train_samples=config.training.max_train_samples,
    )

    tokenizer = create_tokenizer(config.model)
    tokenized = _tokenize_datasets(raw_datasets, tokenizer, config.model.max_length)

    base_model = create_base_sequence_classifier(
        config.model,
        num_labels=num_labels,
        id2label=id2label_map,
        label2id=label2id_map,
    )
    model = apply_lora(base_model, config.lora)
    prepare_model_for_training(model, config.training.gradient_checkpointing)

    comparison: dict[str, Any] = {
        "model_id": config.model.model_id,
        "num_labels": num_labels,
        "baseline": {},
        "after_lora": {},
        "delta": {},
    }

    if config.evaluation.run_baseline_before_training:
        print("Running baseline evaluation (pre–LoRA fine-tuning)...")
        comparison["baseline"] = evaluate_splits(
            model=model,
            tokenized_datasets=tokenized,
            training=config.training,
            evaluation=config.evaluation,
            output_dir=config.paths.output_dir,
            phase="baseline",
        )
        _print_metrics("baseline", comparison["baseline"])

    train_args = TrainingArguments(
        output_dir=str(config.paths.output_dir / "checkpoints"),
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        eval_strategy=config.training.eval_strategy,
        save_strategy=config.training.save_strategy,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        bf16=config.training.bf16 and torch.cuda.is_available(),
        fp16=config.training.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=config.training.gradient_checkpointing,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to=config.training.report_to,
        seed=config.training.seed,
        label_names=["labels"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    adapter_dir = config.paths.output_dir / "lora_adapter"
    trained_model = trainer.model
    if trained_model is None:
        raise RuntimeError("Trainer has no model after training.")
    trained_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir.resolve()}")

    print("Running post–LoRA evaluation...")
    comparison["after_lora"] = evaluate_splits(
        model=trainer.model,
        tokenized_datasets=tokenized,
        training=config.training,
        evaluation=config.evaluation,
        output_dir=config.paths.output_dir,
        phase="after_lora",
    )
    _print_metrics("after_lora", comparison["after_lora"])

    if comparison["baseline"]:
        comparison["delta"] = compare_evaluations(
            comparison["baseline"],
            comparison["after_lora"],
            primary_metric=config.training.metric_for_best_model,
        )
        _print_delta(comparison["delta"])

    comparison_path = config.paths.output_dir / config.evaluation.comparison_filename
    comparison_path.write_text(json.dumps(comparison, indent=2))
    print(f"Wrote evaluation comparison to {comparison_path.resolve()}")

    return comparison


def _print_metrics(phase: str, split_metrics: dict[str, dict[str, float]]) -> None:
    for split, metrics in split_metrics.items():
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1_macro")
        print(f"  [{phase}] {split}: accuracy={acc:.4f} f1_macro={f1:.4f}" if acc is not None else f"  [{phase}] {split}: {metrics}")


def _print_delta(delta: dict[str, dict[str, float]]) -> None:
    for split, metrics in delta.items():
        d_acc = metrics.get("accuracy")
        d_f1 = metrics.get("f1_macro")
        if d_acc is not None and d_f1 is not None:
            print(f"  [delta] {split}: Δaccuracy={d_acc:+.4f} Δf1_macro={d_f1:+.4f}")
