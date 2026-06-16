import yaml
import os
from pathlib import Path


CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    api_key = os.environ.get("NIM_API_KEY", "")
    if api_key:
        config["nim"]["api_key"] = api_key

    return config
