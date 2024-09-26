import json
import logging
from pathlib import Path
from typing import Any


def dump_json(obj: Any, file_path: str | Path):
    num_not_standard_objs = [0]

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            try:
                return super().default(obj)
            except TypeError:
                num_not_standard_objs[0] += 1
                return str(obj)

    Path(file_path).parent.mkdir(exist_ok=True, parents=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=CustomEncoder)
    if num_not_standard_objs[0] > 0:
        logging.warning(
            f"[WARN] {num_not_standard_objs[0]} objs are not serializable and are converted to string format.")
    return Path(file_path)


def load_json(file_path: str | Path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
