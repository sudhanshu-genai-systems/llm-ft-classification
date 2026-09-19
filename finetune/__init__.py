"""Modular BANKING77 LoRA fine-tuning package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from finetune.train_lora import run_lora_finetune


def __getattr__(name: str) -> Any:
    if name == "run_lora_finetune":
        from finetune.train_lora import run_lora_finetune

        return run_lora_finetune
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run_lora_finetune"]
