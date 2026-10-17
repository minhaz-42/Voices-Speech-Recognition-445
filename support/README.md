# support/: the shared code

All project code lives in this folder, plus `tools/` (command-line tools), `main.py` (inference), `tests/` (unit tests) and `configs/config.yaml` (every setting). Folders such as `results/` and `others/` only hold a README until the task that fills them is done.

Run modules from the repository root: `python -m support.<module>`. Run all tests with `python -m unittest discover -s tests -v`.

## How the pieces fit

```text
data/raw/<dataset>/ ──(T42 select_subsets)──► data/subsets/<dataset>.csv
        │
        ▼  support/dataset.py --build            processed 16 kHz clips + manifest
        ▼  tools/make_splits.py                  random split + 5 speaker-independent folds
        ├─► support/classical.py   (features.py)  RF / XGBoost / k-NN / SVM
        ├─► support/train_wav2vec2.py (noise.py)  wav2vec 2.0, untrained or fine-tuned
        └─► support/whisper_asr.py               zero-shot Whisper + WER
                 │
                 ▼  support/evaluate.py          same metrics, reports, confusion matrices for every model
           results/<run>/<dataset>/   ──►  support/noise_test.py, main.py / support/inference.py
```

## Modules merged into `main`

| Module | What it does | Task · owner | Tests |
|---|---|---|---|
| [`config.py`](config.py) | Loads `configs/config.yaml` and turns its relative paths into absolute ones | T01 · M1 | [`test_config.py`](../tests/test_config.py) |
| [`audio.py`](audio.py) | Shared pre-processing: mono, 16 kHz, silence trimmed, peak at −1 dBFS, padded to 6 s and never cut | T01 · M1 | [`test_audio.py`](../tests/test_audio.py) |
| [`metrics.py`](metrics.py) | Accuracy, macro / weighted F1, per-class report, confusion matrix, accuracy per speaker or group | T03 · M3 | [`test_metrics.py`](../tests/test_metrics.py) |
| [`phrases.py`](phrases.py) | Loads and validates a phrase list (extended in T42 for the two datasets' lists) | M2 | [`test_phrases.py`](../tests/test_phrases.py) |
| [`dataset.py`](dataset.py) | Reads a subset list, builds the processed copies and manifest (with SHA-256 checks), serves the train / val / test clips of a split | T08 · M1 | [`test_dataset.py`](../tests/test_dataset.py) |
| [`../tools/make_splits.py`](../tools/make_splits.py) | Random 80/20 split and 5 speaker-independent folds (1 F + 1 M test speakers; same first language on TenVoices-L2), checked for leakage | T08 · M1 | [`test_make_splits.py`](../tests/test_make_splits.py) |
| [`features.py`](features.py) | 78 MFCC summary features per clip (13 MFCC + Δ + ΔΔ, mean and std), cached per dataset | T11 · M1 | [`test_features.py`](../tests/test_features.py) |
| [`evaluate.py`](evaluate.py) | One evaluation for every model: metrics, reports, per-speaker table, confusion matrix; pools the 5 folds | T11 · M1 | [`test_evaluate.py`](../tests/test_evaluate.py) |
| [`classical.py`](classical.py) | Shared runner for Random Forest, XGBoost, k-NN and SVM: hyper-parameters chosen on validation only, scored once on test | T11 · M1 | [`test_classical.py`](../tests/test_classical.py) |
| [`noise.py`](noise.py) | White noise at a chosen SNR; random-SNR augmentation of training clips | T15 · M1 | [`test_noise.py`](../tests/test_noise.py) |
| [`device.py`](device.py) | Picks CUDA, Apple MPS or CPU | T15 · M1 | (via the training tests) |
| [`train_wav2vec2.py`](train_wav2vec2.py) | Fine-tunes or evaluates wav2vec 2.0 (`--epochs`, `--noise`, `--untrained`); best epoch chosen on validation | T15 · M1 | [`test_train_wav2vec2.py`](../tests/test_train_wav2vec2.py) |
| [`whisper_asr.py`](whisper_asr.py) | Zero-shot Whisper speech-to-text, text normalisation, WER / CER, nearest-phrase matching; runs on the splits with WER per speaker and first language | T29, T30 · M1 | [`test_whisper_asr.py`](../tests/test_whisper_asr.py) |
| [`inference.py`](inference.py) | Recognisers for saved models (traditional, wav2vec 2.0, Whisper): raw audio in, top-k phrases out; video and microphone input | T22 · M1 | [`test_inference.py`](../tests/test_inference.py) |
| [`../main.py`](../main.py) | Command-line inference with a benchmark option | T22 · M1 | [`test_inference.py`](../tests/test_inference.py) |
| [`noise_test.py`](noise_test.py) | Accuracy of a saved run with identical white noise at 20 / 10 / 5 dB for every model | T21 · M1 | [`test_noise_test.py`](../tests/test_noise_test.py) |

Modules for later tasks appear here as they are merged; [docs/STATUS.md](../docs/STATUS.md) lists every task with its files.

**macOS note:** PyTorch (wav2vec 2.0, Whisper) and XGBoost must not run in the same Python process: their OpenMP libraries clash and the process crashes or hangs. The command-line tools keep them apart, `classical.py` refuses XGBoost when PyTorch is loaded, and the PyTorch tests run in a child process (`tests/_wav2vec2_check.py`).

## Code that is not merged yet

Each member works on their own branch (`dev_rabbi`, `dev_mahir`, `dev_shafil`) and pushes it at the end of every session, finished or not, so nothing stays only on a laptop. Until Tanvir merges it, that code is not on `main`. To see it:

- **On GitHub:** the **Pull requests** tab (every branch with work has one; a *draft* pull request means a task in it is still unfinished), or pick the branch in the branch selector at the top of the file list.
- **Locally:** `git fetch origin`, then `git diff --stat main...origin/dev_mahir` for the list of changed files and `git diff main...origin/dev_mahir` for the code.
