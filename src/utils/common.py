from pathlib import Path
import yaml

def read_yaml(path: str) -> dict:
    p = Path(path)
    with p.open("r") as f:
        return yaml.safe_load(f)
