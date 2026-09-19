from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SETTINGS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SETTINGS_DIR.parents[1]
DEFAULT_CONFIG_PATH = _SETTINGS_DIR / "config.json"


class LoRABias(str, Enum):
    """PEFT ``LoraConfig.bias`` allowed values."""

    NONE = "none"
    ALL = "all"
    LORA_ONLY = "lora_only"


class PathsSettings(BaseSettings):
    """Filesystem paths for data, caches, and training artifacts."""

    model_config = SettingsConfigDict(extra="ignore")

    data_dir: Path
    output_dir: Path
    cache_dir: Path
    label_mapping_file: str


class ModelSettings(BaseSettings):
    """Hugging Face model and tokenizer options."""

    model_config = SettingsConfigDict(extra="ignore")

    model_id: str
    revision: str | None = None
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    max_length: int = Field(ge=32, le=4096)
    padding_side: str = "right"
    hf_token_env: str = "HF_TOKEN"


class LoRASettings(BaseSettings):
    """PEFT LoRA adapter hyperparameters."""

    model_config = SettingsConfigDict(extra="ignore")

    r: int = Field(ge=1)
    lora_alpha: int = Field(ge=1)
    lora_dropout: float = Field(ge=0.0, le=0.5)
    bias: LoRABias = LoRABias.NONE
    target_modules: list[str]
    modules_to_save: list[str]


class TrainingSettings(BaseSettings):
    """Hugging Face Trainer / training loop hyperparameters."""

    model_config = SettingsConfigDict(extra="ignore")

    num_train_epochs: float = Field(gt=0)
    per_device_train_batch_size: int = Field(ge=1)
    per_device_eval_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    warmup_ratio: float = Field(ge=0, le=1)
    lr_scheduler_type: str
    logging_steps: int = Field(ge=1)
    eval_strategy: str
    save_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    seed: int
    fp16: bool
    bf16: bool
    gradient_checkpointing: bool
    dataloader_num_workers: int = Field(ge=0)
    report_to: list[str]
    max_train_samples: int | None = None


class EvaluationSettings(BaseSettings):
    """Evaluation splits and sampling for before/after LoRA comparison."""

    model_config = SettingsConfigDict(extra="ignore")

    splits: list[str]
    max_eval_samples: int | None = None
    run_baseline_before_training: bool
    comparison_filename: str


class FinetuneConfig(BaseModel):
    """Fine-tuning configuration loaded from ``config.json``."""

    paths: PathsSettings
    model: ModelSettings
    lora: LoRASettings
    training: TrainingSettings
    evaluation: EvaluationSettings

    @model_validator(mode="after")
    def _resolve_paths(self) -> Self:
        root = _repo_root()
        for name in ("data_dir", "output_dir", "cache_dir"):
            value = getattr(self.paths, name)
            if not value.is_absolute():
                setattr(self.paths, name, (root / value).resolve())
        return self

    @classmethod
    def load(cls, config_path: Path | str | None = None) -> FinetuneConfig:
        """Load and validate all settings from ``config.json``."""
        path = Path(config_path) if config_path is not None else _config_path_from_env()
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)


def _repo_root() -> Path:
    return _REPO_ROOT


def _config_path_from_env() -> Path:
    override = os.environ.get("FT_CONFIG_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CONFIG_PATH
