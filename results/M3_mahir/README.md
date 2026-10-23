# results/M3_mahir/: Mahir's results

Mahir Ahmed's (M3) runs are written here by the shared scripts (`result_folders` in `configs/config.yaml`), so they never mix with the other members' runs. None is merged yet.

| Path | What |
|---|---|
| `knn/fsc/` | k-NN on MFCC features (T13) |
| `w2v2_3ep/fsc/` | wav2vec 2.0 fine-tuned 3 epochs (T17) |
| `hubert_15ep_noise/fsc/` | HuBERT Base fine-tuned 15 epochs with white-noise augmentation (T45) |
| `figures/` | Confusion matrices and figures: `python -m support.figures --folder M3_mahir` |

**Every run in detail, with all its numbers: [RUNS.md](RUNS.md)** (written from the files in this folder by `python -m support.contributions`).
