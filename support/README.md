# support/: the shared code

All project code lives in this folder (plus `tools/` for command-line tools, `tests/` for the unit tests and `configs/config.yaml` for the settings). Folders such as `results/`, `notebooks/` and `others/` only hold a README until the task that fills them is done.

Run modules from the repository root: `python -m support.<module>`. Run all tests with `python -m unittest discover -s tests -v`.

## Modules merged into `main`

| Module | What it does | Task · owner | Tests |
|---|---|---|---|
| [`config.py`](config.py) | Loads `configs/config.yaml` and turns its relative paths into absolute ones; `python -m support.config` prints the settings | T01 · M1 | [`test_config.py`](../tests/test_config.py) |
| [`audio.py`](audio.py) | Reads audio and applies the shared pre-processing: mono, 16 kHz, silence trimmed, peak at −1 dBFS, padded to 6 s and never cut; `python -m support.audio clip.wav` | T01 · M1 | [`test_audio.py`](../tests/test_audio.py) |
| [`metrics.py`](metrics.py) | Accuracy, macro / weighted F1, per-class report, confusion matrix (table and plot), accuracy per speaker or per group | T03 · M3 | [`test_metrics.py`](../tests/test_metrics.py) |
| [`phrases.py`](phrases.py) | Loads and validates the phrase list | T02 · M2 | [`test_phrases.py`](../tests/test_phrases.py) |
| [`whisper_asr.py`](whisper_asr.py) | Zero-shot Whisper speech-to-text, text normalisation, WER / CER, nearest-phrase matching and batch runs; `python -m support.whisper_asr clip.wav` | T29 · M1 | [`test_whisper_asr.py`](../tests/test_whisper_asr.py) |

Modules for later tasks appear here as they are merged; [docs/STATUS.md](../docs/STATUS.md) lists every task with its files.

## Code that is not merged yet

Each member works on their own branch (`dev_rabbi`, `dev_mahir`, `dev_shafil`) until Tanvir merges it, so code in progress is not on `main` yet. To see it:

- **On GitHub:** the **Pull requests** tab (every branch with work has one), or pick the branch in the branch selector at the top of the file list.
- **Locally:** `git fetch origin`, then `git diff --stat main...origin/dev_mahir` for the list of changed files and `git diff main...origin/dev_mahir` for the code.
