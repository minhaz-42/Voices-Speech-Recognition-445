"""Load the shared project settings from ``configs/config.yaml``.

Every script reads its settings through :func:`load_config`, and turns the
relative paths in the ``paths`` section into absolute ones with
:func:`resolve_path`. That way the code runs the same from any working
directory, on a laptop or on Colab.

Usage::

    python -m support.config                 # print the settings
    python -m support.config --config other.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

REQUIRED_SECTIONS = ("paths", "audio")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read a YAML config file and check that the required sections exist.

    Args:
        path: Config file to read. Defaults to ``configs/config.yaml``.

    Returns:
        The settings as a nested dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a mapping or a required section is missing.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    missing = [s for s in REQUIRED_SECTIONS if not isinstance(cfg.get(s), dict)]
    if missing:
        raise ValueError(f"{path} is missing the section(s): {', '.join(missing)}")
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Return the absolute path for an entry of the ``paths`` section.

    Args:
        cfg: Settings from :func:`load_config`.
        key: Name of the entry, e.g. ``"raw"`` or ``"phrases"``.

    Returns:
        The path joined to the project root (absolute paths are kept as they are).

    Raises:
        KeyError: If ``key`` is not in the ``paths`` section.
    """
    try:
        value = cfg["paths"][key]
    except KeyError:
        raise KeyError(f"'{key}' is not defined in the paths section of the config") from None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    """Print the settings and the resolved paths."""
    parser = argparse.ArgumentParser(description="Print the TenVoices settings.")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(yaml.safe_dump(cfg, sort_keys=False).rstrip())
    print("\nresolved paths:")
    for key in cfg["paths"]:
        print(f"  {key}: {resolve_path(cfg, key)}")


if __name__ == "__main__":
    main()
