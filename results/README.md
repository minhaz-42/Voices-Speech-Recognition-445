# results/

Small result tables from every run, committed so everyone can continue from them. Nothing is here yet: the first results arrive with **T11** (Random Forest), once the dataset splits exist (**T08**).

- One folder per run, named after the run (e.g. `random_forest/`, `w2v2_15ep_noise/`, `whisper_small/`), with one subfolder per dataset (`fsc/`, `l2arctic/`). The shared scripts in `support/` write `metrics.json`, `predictions.csv`, the classification report and the confusion matrix there.
- `all_results.csv`: every model in one table (T20, M3).
- Only CSV / JSON files under 1 MB; models and audio never go here.

Every number in the README, report, slides and poster must come from a file in this folder.
