"""Dataset statistics computation, reporting, and visualization for TenVoices.

Rabbi (M2) · Task T09
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from support.config import PROJECT_ROOT


def compute_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics from a subset dataframe."""
    stats: Dict[str, Any] = {}

    stats["total_clips"] = len(df)
    stats["unique_speakers"] = int(df["speaker_id"].nunique()) if "speaker_id" in df else 0

    # Gender counts
    if "gender" in df:
        stats["gender_counts"] = df["gender"].value_counts().to_dict()
    else:
        stats["gender_counts"] = {}

    # Duration statistics
    if "duration_s" in df and len(df) > 0:
        stats["duration_min_s"] = float(df["duration_s"].min())
        stats["duration_max_s"] = float(df["duration_s"].max())
        stats["duration_median_s"] = float(df["duration_s"].median())
        stats["duration_mean_s"] = float(df["duration_s"].mean())
        stats["duration_std_s"] = float(df["duration_s"].std()) if len(df) > 1 else 0.0
    else:
        stats["duration_min_s"] = 0.0
        stats["duration_max_s"] = 0.0
        stats["duration_median_s"] = 0.0
        stats["duration_mean_s"] = 0.0
        stats["duration_std_s"] = 0.0

    # QC Flags
    flag_col = "qc_flag" if "qc_flag" in df else ("flags" if "flags" in df else None)
    if flag_col:
        filled_flags = df[flag_col].fillna("ok").replace("", "ok")
        stats["qc_flags"] = filled_flags.value_counts().to_dict()
    else:
        stats["qc_flags"] = {}

    return stats


def generate_dataset_report(dataset: str = "fsc") -> Dict[str, Any]:
    """Load dataset subset and QC data, save stats table and summary figure."""
    data_dir = PROJECT_ROOT / "data"
    subset_csv = data_dir / "subsets" / f"{dataset}.csv"
    qc_csv = data_dir / "subsets" / f"qc_{dataset}.csv"
    prompts_csv = data_dir / "prompts" / f"{dataset}_phrases.csv"

    if not subset_csv.exists():
        raise FileNotFoundError(f"Subset manifest not found: {subset_csv}")

    df = pd.read_csv(subset_csv)

    # Merge phrase category if available
    if "category" not in df.columns and prompts_csv.exists():
        prompts_df = pd.read_csv(prompts_csv)
        if "phrase_id" in prompts_df.columns and "category" in prompts_df.columns:
            cat_map = dict(zip(prompts_df["phrase_id"], prompts_df["category"]))
            df["category"] = df["phrase_id"].map(cat_map)

    if "category" not in df.columns:
        df["category"] = "general"

    # Merge QC flags if available
    if qc_csv.exists():
        qc_df = pd.read_csv(qc_csv)
        if "flags" in qc_df.columns:
            flags_map = dict(zip(qc_df["utt_id"], qc_df["flags"]))
            df["qc_flag"] = df["utt_id"].map(flags_map).fillna("ok").replace("", "ok")

    # Compute overall stats
    stats = compute_dataset_stats(df)

    # 1. Build and save detailed stats CSV
    stats_rows = []
    # Overall summary row
    stats_rows.append({
        "section": "summary",
        "item": "total_clips",
        "value": stats["total_clips"],
        "extra": f"total_duration_min: {df['duration_s'].sum() / 60.0:.2f}",
    })
    stats_rows.append({
        "section": "summary",
        "item": "unique_speakers",
        "value": stats["unique_speakers"],
        "extra": f"genders: {stats['gender_counts']}",
    })
    stats_rows.append({
        "section": "duration",
        "item": "min_median_max_s",
        "value": f"{stats['duration_min_s']:.2f} / {stats['duration_median_s']:.2f} / {stats['duration_max_s']:.2f}",
        "extra": f"mean: {stats['duration_mean_s']:.2f}s, std: {stats['duration_std_s']:.2f}s",
    })
    for flag, count in stats["qc_flags"].items():
        stats_rows.append({
            "section": "qc_flags",
            "item": flag,
            "value": count,
            "extra": f"{count / len(df) * 100:.1f}%",
        })

    # Per-speaker rows
    for spk, spk_df in df.groupby("speaker_id"):
        stats_rows.append({
            "section": "per_speaker",
            "item": spk,
            "value": len(spk_df),
            "extra": f"gender: {spk_df['gender'].iloc[0]}, l1: {spk_df['l1'].iloc[0]}, dur_s: {spk_df['duration_s'].sum():.1f}",
        })

    # Per-category rows
    for cat, cat_df in df.groupby("category"):
        stats_rows.append({
            "section": "per_category",
            "item": cat,
            "value": len(cat_df),
            "extra": f"phrases: {cat_df['phrase_id'].nunique()}",
        })

    stats_df = pd.DataFrame(stats_rows)
    out_csv = data_dir / "subsets" / f"stats_{dataset}.csv"
    stats_df.to_csv(out_csv, index=False)
    print(f"Saved dataset statistics table to: {out_csv}")

    # 2. Draw and save summary figure
    img_dir = PROJECT_ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_img = img_dir / f"dataset_{dataset}.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # Subplot A: Duration distribution
    ax_hist = axes[0]
    ax_hist.hist(df["duration_s"], bins=20, color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax_hist.axvline(
        stats["duration_median_s"],
        color="#d95f02",
        linestyle="--",
        linewidth=2,
        label=f"Median: {stats['duration_median_s']:.2f}s",
    )
    ax_hist.set_title(f"TenVoices-{dataset.upper()}: Clip Durations", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Duration (seconds)")
    ax_hist.set_ylabel("Number of Clips")
    ax_hist.legend(frameon=True)
    ax_hist.grid(axis="y", linestyle=":", alpha=0.6)

    # Subplot B: Clips per category
    ax_cat = axes[1]
    cat_counts = df["category"].value_counts().sort_values(ascending=True)
    cat_counts.plot(kind="barh", ax=ax_cat, color="#388e3c", edgecolor="black", alpha=0.85)
    ax_cat.set_title(f"TenVoices-{dataset.upper()}: Clips per Category", fontsize=11, fontweight="bold")
    ax_cat.set_xlabel("Clip Count")
    ax_cat.grid(axis="x", linestyle=":", alpha=0.6)

    fig.savefig(out_img, dpi=300)
    plt.close(fig)
    print(f"Saved dataset summary figure to: {out_img}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate dataset statistics and figures.")
    parser.add_argument("--dataset", default="fsc", choices=["fsc", "l2arctic"], help="Dataset to process.")
    args = parser.parse_args()

    print(f"Generating statistics for TenVoices-{args.dataset.upper()}...")
    stats = generate_dataset_report(args.dataset)
    print("Summary:")
    print(f"  Total clips: {stats['total_clips']}")
    print(f"  Speakers: {stats['unique_speakers']} ({stats['gender_counts']})")
    print(f"  Duration: min {stats['duration_min_s']:.2f}s | median {stats['duration_median_s']:.2f}s | max {stats['duration_max_s']:.2f}s")
    print(f"  QC Flags: {stats['qc_flags']}")


if __name__ == "__main__":
    main()
