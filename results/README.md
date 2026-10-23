# results/

Small result tables from every run, committed so everyone can continue from them.

- **One folder per member**, each with its runs, figures and a `README.md`, plus `RUNS.md` with every run in detail (model, how it was trained, every split, accuracy per speaker, confusion matrices; written by `python -m support.contributions`):
  - **[`M1_tanvir/`](M1_tanvir/README.md)**: Random Forest, wav2vec 2.0 15 epochs with and without noise, Whisper small; the noise tests (`noise/`) and the speed benchmark.
  - **[`M2_rabbi/`](M2_rabbi/README.md)**: XGBoost, wav2vec 2.0 8 epochs + noise, WavLM Base+ 15 epochs + noise.
  - **[`M3_mahir/`](M3_mahir/README.md)**: k-NN, wav2vec 2.0 3 epochs, HuBERT Base 15 epochs + noise (not merged yet).
  - **[`M4_shafil/`](M4_shafil/README.md)**: SVM, the untrained wav2vec 2.0 baseline, the linear probe.
- `demo_check/`: the demo app checked on a team member's own recordings (T23, Shafil).
- Which run is kept in which member folder is set under `result_folders` in `configs/config.yaml`; the shared scripts in `support/` find the right folder by themselves (`support.evaluate.run_dir`, `list_runs`). Each run folder holds `summary.json`, `predictions_<split>.csv` and `random/`, `folds/` with metrics, classification report, per-speaker accuracy and confusion matrix; the confusion matrix images go to `<member folder>/figures/confusion/`.
- `all_results.csv`: every model in one table (T20, M3).
- Only CSV / JSON files under 1 MB (plus a member folder's figures); models and audio never go here.

Every number in the README, report, slides and poster must come from a file in this folder.
