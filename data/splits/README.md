# data/splits/

Train / validation / test lists, one folder per dataset (`fsc/`, `l2arctic/`): a random 80/20 split and 5 speaker-independent folds each, made by **T08** (M1) with `tools/make_splits.py`. These are small CSV files and are committed, so every model is trained and tested on exactly the same clips.
