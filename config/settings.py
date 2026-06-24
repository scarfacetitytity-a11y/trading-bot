"""Load application configuration from config/config.yaml and .env."""
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML config and .env (for secrets) and return config as a dict."""
    load_dotenv(PROJECT_ROOT / ".env")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Copy config/config.example.yaml to config/config.yaml and edit it."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
