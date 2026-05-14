from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any


REQUIRED_FIELDS = {
    "baseline",
    "stage",
    "dataset",
    "seq_len",
    "pred_len",
    "seed",
    "search_budget",
    "entry",
    "args",
    "env",
    "metric_policy",
    "selection_policy",
}


_ENV_PATTERN = re.compile(r"\$(\w+)|\$\{([^}]+)\}")


def _expand_string(value: str, env_map: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2)
        return env_map.get(key, os.environ.get(key, match.group(0)))

    return _ENV_PATTERN.sub(replace, value)


def _expand_value(value: Any, env_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _expand_string(value, env_map)
    if isinstance(value, list):
        return [_expand_value(item, env_map) for item in value]
    if isinstance(value, dict):
        return {key: _expand_value(item, env_map) for key, item in value.items()}
    return value


@dataclass
class Manifest:
    baseline: str
    stage: str
    dataset: str
    seq_len: int
    pred_len: int | list[int]
    seed: int
    search_budget: dict[str, Any]
    entry: str
    args: dict[str, Any]
    env: dict[str, str]
    metric_policy: str
    selection_policy: str
    path: Path
    raw: dict[str, Any]

    @property
    def repo_relative_entry(self) -> Path:
        return Path(self.entry)

    @property
    def pred_lens(self) -> list[int]:
        return self.pred_len if isinstance(self.pred_len, list) else [self.pred_len]


def _normalize_pred_len(value: Any) -> int | list[int]:
    if isinstance(value, bool):
        raise TypeError("Manifest field 'pred_len' must be an int or list[int], not bool.")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("Manifest field 'pred_len' must be positive.")
        return value
    if isinstance(value, list):
        if not value:
            raise ValueError("Manifest field 'pred_len' cannot be an empty list.")
        normalized: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError("Manifest field 'pred_len' list must contain only integers.")
            if item <= 0:
                raise ValueError("Manifest field 'pred_len' list must contain only positive integers.")
            normalized.append(item)
        if len(set(normalized)) != len(normalized):
            raise ValueError("Manifest field 'pred_len' list cannot contain duplicates.")
        return normalized
    raise TypeError("Manifest field 'pred_len' must be an int or list[int].")


def manifest_from_data(data: dict[str, Any], manifest_path: str | Path) -> Manifest:
    resolved_path = Path(manifest_path).resolve()
    missing = sorted(REQUIRED_FIELDS.difference(data))
    if missing:
        raise ValueError(f"Manifest missing required fields: {', '.join(missing)}")
    env_map = {str(key): os.path.expandvars(str(value)) for key, value in data.get("env", {}).items()}
    expanded = _expand_value(data, env_map)
    if not isinstance(expanded["args"], dict):
        raise TypeError("Manifest field 'args' must be an object.")
    if not isinstance(expanded["env"], dict):
        raise TypeError("Manifest field 'env' must be an object.")
    return Manifest(
        baseline=str(expanded["baseline"]),
        stage=str(expanded["stage"]),
        dataset=str(expanded["dataset"]),
        seq_len=int(expanded["seq_len"]),
        pred_len=_normalize_pred_len(expanded["pred_len"]),
        seed=int(expanded["seed"]),
        search_budget=dict(expanded["search_budget"]),
        entry=str(expanded["entry"]),
        args=dict(expanded["args"]),
        env={str(key): str(value) for key, value in expanded["env"].items()},
        metric_policy=str(expanded["metric_policy"]),
        selection_policy=str(expanded["selection_policy"]),
        path=resolved_path,
        raw=expanded,
    )


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_from_data(data, manifest_path)
