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


CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]
CHECKLIST_START = "<!-- numbers checklist: written by python -m support.presentation -->"
CHECKLIST_END = "<!-- end of numbers checklist -->"


def find_chrome() -> str | None:
    """Path of Chrome, Chromium or Edge, or ``None`` when none is installed."""
    import shutil

    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge"):
        if shutil.which(name):
            return shutil.which(name)
    return next((p for p in CHROME_CANDIDATES if Path(p).exists()), None)


def render(html: Path, out: Path, width: int, height: int, chrome: str | None = None) -> None:
    """Render a local HTML page with headless Chrome: PNG at 2x scale, or PDF when ``out`` ends in .pdf.

    Raises:
        RuntimeError: If no Chrome / Edge is installed.
    """
    import subprocess

    chrome = chrome or find_chrome()
    if chrome is None:
        raise RuntimeError("no Chrome or Edge found: install one, or use --no-render")
    out.parent.mkdir(parents=True, exist_ok=True)
    url = Path(html).resolve().as_uri()
    common = [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=3000"]
    if out.suffix == ".pdf":
        args = common + ["--no-pdf-header-footer", f"--print-to-pdf={out.resolve()}", url]
    else:
        args = common + ["--force-device-scale-factor=2", f"--window-size={width},{height}",
                         f"--screenshot={out.resolve()}", url]
    subprocess.run(args, check=True, capture_output=True, timeout=120)


def checklist_markdown(used: dict[str, list[dict[str, Any]]]) -> str:
    """The numbers checklist: per output file, every placeholder, its value and its source."""
    lines = [CHECKLIST_START, "", "### Numbers checklist (generated, do not edit by hand)", ""]
    for name, rows in used.items():
        lines += [f"**{name}**", "", "| Placeholder | Value | Source |", "|---|---|---|"]
        lines += [f"| `{r['key']}` | {r['value']} | {('`' + r['source'] + '`') if r['source'] else 'not merged yet'} |"
                  for r in rows]
        lines.append("")
    lines.append(CHECKLIST_END)
    return "\n".join(lines)


def update_checklist(readme: Path, used: dict[str, list[dict[str, Any]]]) -> None:
    """Write the checklist into ``readme`` between its markers (appended the first time)."""
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    block = checklist_markdown(used)
    if CHECKLIST_START in text and CHECKLIST_END in text:
        start, end = text.index(CHECKLIST_START), text.index(CHECKLIST_END) + len(CHECKLIST_END)
        text = text[:start] + block + text[end:]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    readme.write_text(text, encoding="utf-8", newline="\n")


# Template -> (filled HTML, rendered output, CSS width, CSS height)
OUTPUTS = {
    "poster_results.template.html": ("poster_results.html", "images/poster_results.png", 1600, 1200),
    "slides.template.html": ("slides.html", "others/slides.pdf", 1280, 720),
}


def build(results: str | Path | None = None, others: str | Path = "others", do_render: bool = True,
          chrome: str | None = None) -> list[Path]:
    """Fill every template in ``others``, update the checklist and (optionally) render the outputs.

    Returns:
        The files written.
    """
    others = PROJECT_ROOT / others if not Path(others).is_absolute() else Path(others)
    values = collect_values(results)
    written: list[Path] = []
    used: dict[str, list[dict[str, Any]]] = {}
    for template, (html_name, rendered, width, height) in OUTPUTS.items():
        source = others / template
        if not source.exists():
            continue
        text, rows = fill(source.read_text(encoding="utf-8"), values)
        html = others / html_name
        html.write_text(text, encoding="utf-8", newline="\n")
        written.append(html)
        used[f"{html_name} → {rendered}"] = rows
        if do_render:
            out = PROJECT_ROOT / rendered
            render(html, out, width, height, chrome)
            written.append(out)
    if used:
        update_checklist(others / "README.md", used)
        written.append(others / "README.md")
    return written


def main() -> None:
    """Fill the poster and slide templates from ``results/`` and render them."""
    import argparse

    parser = argparse.ArgumentParser(description="Fill the poster and slides with numbers from results/.")
    parser.add_argument("--results", default=None, help="results folder (default: config paths.results)")
    parser.add_argument("--no-render", action="store_true", help="only write the filled HTML and the checklist")
    parser.add_argument("--chrome", default=None, help="path of Chrome / Edge (default: found automatically)")
    args = parser.parse_args()
    for path in build(args.results, do_render=not args.no_render, chrome=args.chrome):
        print(f"saved: {_relative(path)}")


if __name__ == "__main__":
    main()
