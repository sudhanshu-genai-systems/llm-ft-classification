"""
model_factory.py
----------------
Load Llama 3B from Hugging Face and wrap with PEFT LoRA for sequence classification.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from finetune.settings.app import LoRASettings, ModelSettings
except ImportError:
    from settings.app import LoRASettings, ModelSettings


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[key]


def _hf_token(settings: ModelSettings) -> str | None:
    token = os.environ.get(settings.hf_token_env)
    if token:
        return token
    return os.environ.get("HUGGING_FACE_HUB_TOKEN")


def create_tokenizer(settings: ModelSettings):
    token = _hf_token(settings)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_id,
        revision=settings.revision,
        trust_remote_code=settings.trust_remote_code,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = settings.padding_side
    return tokenizer


def create_base_sequence_classifier(
    settings: ModelSettings,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> Any:
    token = _hf_token(settings)
    dtype = _resolve_dtype(settings.torch_dtype)

    quant_config = None
    if settings.load_in_4bit or settings.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=settings.load_in_4bit,
            load_in_8bit=settings.load_in_8bit,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model_kwargs: dict[str, Any] = {
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
        "trust_remote_code": settings.trust_remote_code,
        "token": token,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForSequenceClassification.from_pretrained(
        settings.model_id,
        revision=settings.revision,
        **model_kwargs,
    )
    if model.config.pad_token_id is None and hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = model.config.eos_token_id
    return model


def apply_lora(model: Any, lora: LoRASettings) -> Any:
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        bias=lora.bias.value,
        target_modules=lora.target_modules,
        modules_to_save=lora.modules_to_save,
    )
    return get_peft_model(model, config)


def prepare_model_for_training(model: Any, gradient_checkpointing: bool) -> None:
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
