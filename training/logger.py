"""Logger JSONL simple : une ligne par événement de log (pas de dépendance à
un service externe type W&B, pour rester reproductible hors-ligne)."""
import json
import time
from pathlib import Path


class JSONLLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")

    def log(self, **kwargs):
        record = {"ts": time.time(), **kwargs}
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()
