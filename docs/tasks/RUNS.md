# How to run a model on TenVoices-FSC

Every run task (T12–T19, T44–T46) uses the shared code and the merged subsets and splits, so nobody waits for anyone. Steps 1–2 are needed once per computer.

## 1. Get the audio (once)

The audio is never in git. Either copy the archive from the team Drive (`TenVoices/raw/fsc/fluent.tar.gz`) or download the same file yourself (URL and SHA-256 in [data/SOURCES.md](../../data/SOURCES.md)):

```bash
mkdir -p data/raw/fsc
curl -L -o data/raw/fsc/fluent.tar.gz "https://huggingface.co/datasets/leo19941227/fluent_speech_commands/resolve/main/fluent.tar.gz"
shasum -a 256 data/raw/fsc/fluent.tar.gz   # must print c9fd67f2efa078daa84daddcad2de937eb96581c140e3131ed8cd06fbae9ba1b
tar -xzf data/raw/fsc/fluent.tar.gz -C data/raw/fsc
```

(Windows: `certutil -hashfile data\raw\fsc\fluent.tar.gz SHA256`, and extract with `tar -xzf` in PowerShell.)

## 2. Build the processed copies (once)

```bash
python -m support.dataset --dataset fsc --build
```

This checks every one of the 1,000 selected clips against its SHA-256 in `data/subsets/fsc.csv`, so you know you have exactly the team's data, and writes the 16 kHz copies to `data/processed/fsc/` (gitignored).

## 3. Run

| Kind | Command | Where |
|---|---|---|
| Traditional model | `python -m support.classical --model <random_forest\|xgboost\|knn\|svm> --dataset fsc` | any laptop, minutes |
| wav2vec 2.0 family | `python -m support.train_wav2vec2 --dataset fsc --epochs <N> [--noise white] [--model <checkpoint>]` | a GPU: Colab with [notebooks/train_wav2vec2.ipynb](../../notebooks/train_wav2vec2.ipynb), or an Apple Silicon Mac (slower) |
| Untrained baseline | `python -m support.train_wav2vec2 --dataset fsc --untrained` | any laptop |

- By default a run covers the random split and all 5 folds (`--split all`). Try a quick check first: `--split random --epochs 1 --limit 40`.
- On macOS, PyTorch (wav2vec 2.0 family) and XGBoost must not run in the same Python process. Separate commands are fine.

## 4. What to commit

- `results/<run>/fsc/` (CSV and JSON, small) and `images/confusion/<run>_fsc_*.png`.
- **Not** `outputs/` (features, models: gitignored). Upload trained wav2vec 2.0-family models to the Drive (`TenVoices/models/<run>/`), as the Colab notebook does.
- In your progress file, the faculty update quotes the numbers from `results/<run>/fsc/summary.json` (`random.accuracy` and `folds.pooled.accuracy`), never from memory.

TenVoices-L2 runs use the same commands with `--dataset l2arctic`; M1 runs them for every merged model in T47.
