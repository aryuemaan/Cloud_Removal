"""
Configuration loading utilities.

Loads the master YAML config and exposes it as an attribute-accessible
object so downstream modules can do `cfg.training.batch_size` instead of
`cfg["training"]["batch_size"]`.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigDict(dict):
    """Dict subclass that also allows attribute-style access, recursively."""

    def __init__(self, d: Dict[str, Any] | None = None):
        super().__init__()
        d = d or {}
        for k, v in d.items():
            self[k] = self._wrap(v)

    @staticmethod
    def _wrap(v):
        if isinstance(v, dict):
            return ConfigDict(v)
        if isinstance(v, list):
            return [ConfigDict(x) if isinstance(x, dict) else x for x in v]
        return v

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, ConfigDict) else x for x in v]
            else:
                out[k] = v
        return out


def load_config(path: str | os.PathLike) -> ConfigDict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ConfigDict(raw)


def merge_overrides(cfg: ConfigDict, overrides: Dict[str, Any]) -> ConfigDict:
    """Merge dotted-key overrides, e.g. {'training.batch_size': 8}, into cfg."""
    cfg = ConfigDict(copy.deepcopy(cfg.to_dict()))
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return cfg


def default_config_path() -> str:
    here = Path(__file__).resolve().parents[2]
    return str(here / "config" / "config.yaml")
