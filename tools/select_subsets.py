"""Select the two TenVoices subsets from the downloaded datasets, by the rules in docs/DATASETS.md §2.

Selection is code, not hand-picking: the same source data and seed always give the same
files. A clip is **eligible** when it is readable, mono, at the expected sample rate and
0.5–6 s long after the shared pre-processing without padding (silence trimmed).

**TenVoices-FSC** (Fluent Speech Commands): phrase text = transcription lower-cased,
punctuation removed, spaces collapsed. The 50 phrasings with the most speakers having two
eligible takes are kept (ties alphabetically) and numbered FSC01–FSC50 in alphabetical
order. The 5 female and 5 male speakers with two eligible takes of the most kept phrases
are chosen (ties broken with ``numpy.random.default_rng(seed)``), and their first two
eligible takes (by file name) of every kept phrase are used; a missing take stays missing.
Category = the intent's action.

**TenVoices-L2** (L2-ARCTIC): the 10 fixed speakers; the prompts that all 10 have with
identical text and an eligible clip, sorted, then 50 drawn with ``default_rng(seed)``.

Output (committed): ``data/prompts/<fsc_phrases|l2arctic_sentences>.csv`` and
``data/subsets/<dataset>.csv``. Eligibility checks are cached in ``outputs/select/``.

Usage::

    python -m tools.select_subsets --dataset fsc
    python -m tools.select_subsets --dataset all
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from support.audio import load_audio
from support.config import PROJECT_ROOT, load_config, resolve_path
from support.dataset import SUBSET_COLUMNS, dataset_config, sha256_file

FSC_ROOT = "fluent_speech_commands_dataset"
FSC_RATE, L2_RATE = 16000, 44100
MIN_S, MAX_S = 0.5, 6.0
L2_SPEAKERS = {  # speaker: (gender, first language), the fixed TenVoices-L2 design (DATASETS.md §2)
    "ASI": ("male", "Hindi"), "RRBI": ("male", "Hindi"), "SVBI": ("female", "Hindi"), "TNI": ("female", "Hindi"),
    "ABA": ("male", "Arabic"), "SKA": ("female", "Arabic"), "BWC": ("male", "Mandarin"),
    "LXC": ("female", "Mandarin"), "EBVS": ("male", "Spanish"), "NJS": ("female", "Spanish"),
}


def _abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalise_text(text: str) -> str:
    """Lower case, punctuation removed, spaces collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(text).lower())).strip()


def check_clip(path: Path, expected_rate: int, audio_cfg: dict[str, Any]) -> dict[str, Any]:
    """Eligibility of one source clip: readable, mono, expected rate, 0.5–6 s after trimming."""
    try:
        info = sf.info(str(path))
        if info.channels != 1 or info.samplerate != expected_rate:
            return {"eligible": False, "duration_s": float("nan"), "reason": f"{info.channels} ch, {info.samplerate} Hz"}
        duration = load_audio(path, audio_cfg, pad=False).size / int(audio_cfg["sample_rate"])
    except Exception as err:  # noqa: BLE001 - any unreadable file is simply not eligible
        return {"eligible": False, "duration_s": float("nan"), "reason": f"unreadable: {type(err).__name__}"}
    ok = MIN_S <= duration <= MAX_S
    return {"eligible": ok, "duration_s": round(duration, 3), "reason": "" if ok else f"{duration:.2f} s after trimming"}


def eligibility(paths: list[Path], expected_rate: int, cache: Path | None, audio_cfg: dict[str, Any]) -> pd.DataFrame:
    """Check many clips, reusing a cache keyed by path and file size."""
    known = pd.DataFrame()
    if cache is not None and cache.exists():
        known = pd.read_csv(cache, keep_default_na=False, na_values=[""]).set_index("path")
    rows = []
    for path in paths:
        key, size = str(path), path.stat().st_size if path.exists() else -1
        if key in known.index and int(known.loc[key, "size"]) == size:
            rows.append({"path": key, "size": size, **{k: known.loc[key, k] for k in ("eligible", "duration_s", "reason")}})
        else:
            rows.append({"path": key, "size": size, **check_clip(path, expected_rate, audio_cfg)})
    table = pd.DataFrame(rows)
    table["eligible"] = table["eligible"].astype(str).str.lower().isin(["true", "1"])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache, index=False)
    return table.set_index("path")


def _most_common(values: pd.Series) -> str:
    """The most frequent value; ties go to the alphabetically first."""
    counts = values.value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def _write(subset: pd.DataFrame, phrases: pd.DataFrame, dataset: str, cfg: dict[str, Any]) -> None:
    dcfg = dataset_config(dataset, cfg)
    for frame, target in ((subset[SUBSET_COLUMNS], dcfg["subset"]), (phrases, dcfg["phrases"])):
        path = _abs(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)


def select_fsc(cfg: dict[str, Any] | None = None, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select TenVoices-FSC. Returns ``(subset, phrases)``."""
    cfg = cfg if cfg is not None else load_config()
    dcfg = dataset_config("fsc", cfg)
    seed, n_phrases, per_gender = int(cfg.get("seed", 42)), int(dcfg["n_phrases"]), int(dcfg["speakers_per_gender"])
    raw = _abs(dcfg["raw"])
    root = raw / FSC_ROOT
    clips = pd.concat([pd.read_csv(root / "data" / f"{p}_data.csv") for p in ("train", "valid", "test")],
                      ignore_index=True)
    demo = pd.read_csv(root / "data" / "speaker_demographics.csv").set_index("speakerId")
    clips["text"] = clips["transcription"].map(normalise_text)
    clips["file"] = clips["path"].map(lambda p: Path(p).name)
    clips["abs"] = clips["path"].map(lambda p: root / p)
    cache = resolve_path(cfg, "outputs") / "select" / "fsc_eligibility.csv"
    elig = eligibility(list(clips["abs"]), FSC_RATE, cache, cfg["audio"])
    clips["eligible"] = clips["abs"].map(lambda p: bool(elig.loc[str(p), "eligible"]))
    clips["duration_s"] = clips["abs"].map(lambda p: elig.loc[str(p), "duration_s"])

    good = clips[clips["eligible"]].sort_values(["speakerId", "text", "file"])
    takes = good.groupby(["speakerId", "text"]).size()
    two = takes[takes >= 2].reset_index()[["speakerId", "text"]]

    per_text = two.groupby("text")["speakerId"].nunique().reset_index(name="n")
    kept = sorted(per_text.sort_values(["n", "text"], ascending=[False, True]).head(n_phrases)["text"])
    phrase_id = {t: f"FSC{i:02d}" for i, t in enumerate(kept, start=1)}

    counts = two[two["text"].isin(kept)].groupby("speakerId").size()
    speakers = pd.DataFrame({"speaker_id": sorted(demo.index.astype(str))})
    speakers["n"] = speakers["speaker_id"].map(counts).fillna(0).astype(int)
    speakers["tiebreak"] = np.random.default_rng(seed).random(len(speakers))
    speakers["gender"] = speakers["speaker_id"].map(demo["gender"].str.strip().str.lower())
    chosen = []
    for gender in ("female", "male"):
        pool = speakers[speakers["gender"] == gender].sort_values(["n", "tiebreak"], ascending=[False, True])
        if len(pool) < per_gender:
            raise ValueError(f"only {len(pool)} {gender} speakers in the demographics file")
        chosen += list(pool["speaker_id"].head(per_gender))

    action = clips[clips["text"].isin(kept)].groupby("text")["action"].agg(_most_common)
    rows = []
    for spk in chosen:
        mine = good[(good["speakerId"] == spk) & good["text"].isin(kept)]
        for text, group in mine.groupby("text"):
            for take, clip in enumerate(group.sort_values("file").head(2).itertuples(), start=1):
                pid = phrase_id[text]
                rows.append({"utt_id": f"fsc_{spk}_{pid}_t{take}", "dataset": "fsc", "speaker_id": spk,
                             "gender": demo.loc[spk, "gender"].strip().lower(),
                             "l1": str(demo.loc[spk, "First Language spoken"]).strip(), "phrase_id": pid,
                             "text": text, "take": take, "source_path": Path(clip.abs).relative_to(raw).as_posix(),
                             "duration_s": clip.duration_s, "sha256": sha256_file(clip.abs)})
    subset = pd.DataFrame(rows).sort_values(["speaker_id", "phrase_id", "take"]).reset_index(drop=True)
    phrases = pd.DataFrame({"phrase_id": [phrase_id[t] for t in kept], "category": [action[t] for t in kept],
                            "text": kept})
    if write:
        _write(subset, phrases, "fsc", cfg)
    return subset, phrases


def _find_speaker_dir(raw: Path, speaker: str) -> Path:
    """``<raw>/<SPK>`` or ``<raw>/<any folder>/<SPK>`` (archives differ in their top folder)."""
    for candidate in [raw / speaker, *raw.glob(f"*/{speaker}")]:
        if (candidate / "wav").is_dir():
            return candidate
    raise FileNotFoundError(f"speaker folder {speaker}/wav not found under {raw}")


def select_l2arctic(cfg: dict[str, Any] | None = None, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select TenVoices-L2. Returns ``(subset, phrases)``."""
    cfg = cfg if cfg is not None else load_config()
    dcfg = dataset_config("l2arctic", cfg)
    seed, n_phrases = int(cfg.get("seed", 42)), int(dcfg["n_phrases"])
    raw = _abs(dcfg["raw"])
    speakers = list(dcfg.get("speakers") or L2_SPEAKERS)
    texts: dict[str, dict[str, str]] = {}
    wavs: dict[str, dict[str, Path]] = {}
    for spk in speakers:
        folder = _find_speaker_dir(raw, spk)
        wavs[spk] = {p.stem: p for p in sorted((folder / "wav").glob("*.wav"))}
        texts[spk] = {p.stem: (folder / "transcript" / f"{p.stem}.txt").read_text(encoding="utf-8").strip()
                      for p in wavs[spk].values() if (folder / "transcript" / f"{p.stem}.txt").exists()}
    common = sorted(set.intersection(*(set(texts[s]) for s in speakers)))
    common = [pid for pid in common if len({texts[s][pid] for s in speakers}) == 1]
    cache = resolve_path(cfg, "outputs") / "select" / "l2arctic_eligibility.csv"
    elig = eligibility([wavs[s][pid] for s in speakers for pid in common], L2_RATE, cache, cfg["audio"])
    common = [pid for pid in common if all(bool(elig.loc[str(wavs[s][pid]), "eligible"]) for s in speakers)]
    if len(common) < n_phrases:
        raise ValueError(f"only {len(common)} prompts are usable for all speakers")
    chosen = sorted(np.random.default_rng(seed).choice(common, size=n_phrases, replace=False).tolist())

    rows = []
    for spk in speakers:
        gender, l1 = L2_SPEAKERS[spk]
        for pid in chosen:
            path = wavs[spk][pid]
            rows.append({"utt_id": f"l2_{spk}_{pid}", "dataset": "l2arctic", "speaker_id": spk, "gender": gender,
                         "l1": l1, "phrase_id": pid, "text": texts[spk][pid], "take": 1,
                         "source_path": path.relative_to(raw).as_posix(),
                         "duration_s": elig.loc[str(path), "duration_s"], "sha256": sha256_file(path)})
    subset = pd.DataFrame(rows).sort_values(["speaker_id", "phrase_id"]).reset_index(drop=True)
    phrases = pd.DataFrame({"phrase_id": chosen, "category": "sentence", "text": [texts[speakers[0]][p] for p in chosen]})
    if write:
        _write(subset, phrases, "l2arctic", cfg)
    return subset, phrases


SELECTORS = {"fsc": select_fsc, "l2arctic": select_l2arctic}


def main() -> None:
    """Select one or both subsets and print a summary."""
    parser = argparse.ArgumentParser(description="Select the TenVoices subsets from the downloaded datasets.")
    parser.add_argument("--dataset", required=True, choices=[*SELECTORS, "all"])
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for name in (SELECTORS if args.dataset == "all" else [args.dataset]):
        subset, phrases = SELECTORS[name](cfg)
        genders = subset.groupby("gender")["speaker_id"].nunique().to_dict()
        print(f"{name}: {len(subset)} clips, {subset['speaker_id'].nunique()} speakers {genders}, "
              f"{len(phrases)} phrases, {subset['duration_s'].astype(float).sum() / 60:.1f} min of speech")


if __name__ == "__main__":
    main()
