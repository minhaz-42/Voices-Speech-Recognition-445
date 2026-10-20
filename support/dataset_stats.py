"""Dataset statistics computation and reporting for TenVoices.

Rabbi (M2) · Task T09
"""

from typing import Any, Dict
import pandas as pd


def compute_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics from a subset dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing at least: 'utt_id', 'speaker_id', 'gender',
        'duration_s', and optionally 'category', 'phrase_id', 'qc_flag' / 'flags'.

    Returns
    -------
    dict
        Dictionary containing counts, gender breakdown, duration stats,
        and QC flag counts.
    """
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
        # Fill missing/empty flags with 'ok'
        filled_flags = df[flag_col].fillna("ok").replace("", "ok")
        stats["qc_flags"] = filled_flags.value_counts().to_dict()
    else:
        stats["qc_flags"] = {}

    return stats
