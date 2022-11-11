from pathlib import Path

from omegaconf import DictConfig


def validate_config(config: DictConfig) -> bool:
    pass


def get_config(config_path: Path) -> DictConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"config : {config_path}")

    config = getattr(config_path, "config")
    if not validate_config(config):
        raise ValueError("config is invalid.")

    return config
