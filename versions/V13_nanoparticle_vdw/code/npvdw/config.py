"""JSON-backed parameter sets.  Literature and calibrated sets never mix."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from .potential import VdwParameters

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_FIELDS = {f.name for f in fields(VdwParameters)}


def load_parameters(name_or_path) -> VdwParameters:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / ("%s.json" % path.name)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path.name
    data = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(data) - _FIELDS
    if unknown:
        raise ValueError("unknown keys in %s: %s" % (path, sorted(unknown)))
    return VdwParameters(**data)


def save_parameters(params: VdwParameters, path) -> Path:
    path = Path(path)
    if not path.suffix:
        path = CONFIG_DIR / ("%s.json" % path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(params.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def available_parameters() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))
