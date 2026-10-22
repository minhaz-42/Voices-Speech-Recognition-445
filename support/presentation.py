"""Fill the poster and slide templates with numbers from ``results/`` (task T27).

Templates in ``others/`` (``*.template.html``) contain placeholders instead of numbers,
so no number is ever typed by hand and every one can be traced to its file:

- ``{{fsc:svm:unseen:accuracy}}``: accuracy of a run on unseen speakers (5 folds pooled)
  or on the ``random`` split, as a percentage; ``macro_f1`` is shown as 0.000;
- ``{{fsc:noise:w2v2_15ep_noise:5}}``: unseen-speaker accuracy at 5 dB SNR (``clean`` too);
- ``{{fsc:wer:whisper_small}}``: Whisper's word error rate on the unseen-speaker folds;
- ``{{demo:whisper_small}}``: correct / total in the live check of my own voice.

A placeholder whose run is not merged yet becomes ``TBD``.

Usage::

    python -m support.presentation [--results results] [--no-render]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from support.config import PROJECT_ROOT, load_config, resolve_path

PLACEHOLDER = re.compile(r"\{\{([a-z0-9_:.]+)\}\}")
TBD = "TBD"


def _relative(path: Path) -> str:
    """``path`` relative to the project root, with forward slashes."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def collect_values(results: str | Path | None = None) -> dict[str, tuple[str, str]]:
    """Every number a template can use, formatted, with the file it comes from.

    Args:
        results: Results folder (default: ``paths.results`` of the config).

    Returns:
        ``{key: (text, source file)}`` for the placeholders described in the module docstring.
    """
    from support.evaluate import list_runs
    from support.visualization import noise_table

    cfg = load_config()
    if results is not None:
        cfg["paths"]["results"] = str(Path(results).resolve())
    values: dict[str, tuple[str, str]] = {}

    for run in list_runs(None, cfg).itertuples():
        path = Path(run.path) / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        source = _relative(path)
        for protocol, block in (("random", summary.get("random")),
                                ("unseen", (summary.get("folds") or {}).get("pooled"))):
            if block:
                values[f"{run.dataset}:{run.run}:{protocol}:accuracy"] = (f"{block['accuracy']:.1%}", source)
                values[f"{run.dataset}:{run.run}:{protocol}:macro_f1"] = (f"{block['macro_f1']:.3f}", source)
        wer = (summary.get("error_rates_pooled_folds") or {}).get("wer")
        if wer is not None:
            values[f"{run.dataset}:wer:{run.run}"] = (f"{wer:.1%}", source)

    root = resolve_path(cfg, "results")
    for dataset in ("fsc", "l2arctic"):
        for row in noise_table(root, dataset).itertuples():
            source = next((_relative(p) for p in root.rglob(f"noise/{row.model}_{dataset}.csv")), "")
            values[f"{dataset}:noise:{row.model}:{row.snr_db}"] = (f"{float(row.accuracy):.1%}", source)

    for path in sorted(root.glob("demo_check/*.csv")):
        table = pd.read_csv(path)
        for name, rows in table.groupby("recognizer"):
            correct = int(rows["correct"].astype(str).str.lower().eq("true").sum())
            values[f"demo:{name}"] = (f"{correct}/{len(rows)}", _relative(path))
    return values


def fill(template: str, values: dict[str, tuple[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    """Replace every placeholder in ``template``.

    Returns:
        The filled text and one ``{"key", "value", "source"}`` per placeholder used, in
        order of first use; unknown keys become ``TBD`` with an empty source.
    """
    used: dict[str, dict[str, Any]] = {}

    def replace(match: re.Match) -> str:
        key = match.group(1)
        text, source = values.get(key, (TBD, ""))
        used.setdefault(key, {"key": key, "value": text, "source": source})
        return text

    return PLACEHOLDER.sub(replace, template), list(used.values())
