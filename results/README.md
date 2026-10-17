# results/

Small result tables from every run, committed so everyone can continue from them.

- **`M1_tanvir/`**: everything of Tanvir's in one folder: the runs `random_forest/`, `w2v2_15ep_noise/`, `w2v2_15ep/` and `whisper_small/`, the noise tests (`noise/`), the speed benchmark and all figures (`figures/`). Index: [M1_tanvir/README.md](M1_tanvir/README.md).
- **Other runs:** one folder per run, named after the run (e.g. `xgboost/`, `knn/`, `svm/`), with one subfolder per dataset (`fsc/`, `l2arctic/`); their confusion matrices go to `images/confusion/`.
- Which runs are kept in a member folder is set under `result_folders` in `configs/config.yaml`; the shared scripts in `support/` find the right folder by themselves (`support.evaluate.run_dir`, `list_runs`). Each run folder holds `summary.json`, `predictions_<split>.csv` and `random/`, `folds/` with metrics, classification report, per-speaker accuracy and confusion matrix.
- `all_results.csv`: every model in one table (T20, M3).
- Only CSV / JSON files under 1 MB (plus a member folder's figures); models and audio never go here.

Every number in the README, report, slides and poster must come from a file in this folder.
