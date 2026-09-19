#!/usr/bin/env python3
"""
Entry point for LoRA fine-tuning (local machine or SageMaker training job).

Hyperparameters are read from ``finetune/settings/config.json`` (override path with ``--config``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv(_REPO_ROOT / ".ENV")

    parser = argparse.ArgumentParser(description="LoRA fine-tune Llama 3B on BANKING77")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.json (default: finetune/settings/config.json)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Processed splits directory (overrides config.json paths.data_dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Training output directory (overrides config.json paths.output_dir)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training rows for smoke tests",
    )
    args = parser.parse_args()

    from finetune.settings import FinetuneConfig
    from finetune.train_lora import run_lora_finetune

    config = FinetuneConfig.load(args.config)
    if args.data_dir is not None:
        config.paths.data_dir = args.data_dir
    if args.output_dir is not None:
        config.paths.output_dir = args.output_dir
    if args.max_train_samples is not None:
        config.training.max_train_samples = args.max_train_samples

    run_lora_finetune(config)


if __name__ == "__main__":
    main()
