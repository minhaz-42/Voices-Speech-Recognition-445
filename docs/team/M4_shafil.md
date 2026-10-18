# M4 · Shafil Ahmed

**In short:** Makes the results visible (figures, demo app, poster, slides, video) and runs SVM, the untrained baseline and the linear probe.

On this page: [My tasks](#my-tasks) · [Faculty update](#faculty-update) · [Task cards](#task-cards) · back to the [team index](README.md) or the [task board](../STATUS.md)

| | |
|---|---|
| **Role** | Visualisation, demo app, poster, slides, video |
| **Models** | SVM (traditional) · untrained wav2vec 2.0 baseline · **linear probe** on frozen wav2vec 2.0 |
| **Branch** | `dev_shafil` (Tanvir merges it into `main`) |
| **Owns** | `support/visualization.py`, `support/embeddings.py`, `app.py`, `others/poster.html`, `images/poster.png`, slides and video in `others/`, `docs/literature/self_supervised_speech.md`, `results/svm/`, `results/w2v2_untrained/`, `results/w2v2_probe/` |
| **Report** | Section 7 (demo system) |

**Who edits what:** Shafil (or Shafil's assistant) edits *My tasks*, *Faculty update*, *Requests for the lead* and *Handoff log*. Tanvir writes the notes and the *Task cards* section at the end. Dependencies and merge status: [STATUS.md](../STATUS.md).

## Note from Tanvir (2026-10-17, evening): nothing to wait for

Every one of your tasks below depends only on work that is already merged or on your own earlier tasks. Data preparation (download, selection, QC, splits) is done and merged for TenVoices-FSC, so you can run your models today: [docs/RUNS.md](../RUNS.md) shows how. Work that needs everyone's results is split: you write and test the code, and Tanvir runs it on all results at the end.

## First-time setup

1. You already have write access on GitHub (`Shafil10`), and the branch `dev_shafil` exists.
2. `git clone https://github.com/minhaz-42/Voices-Speech-Recognition-445.git`, then `cd Voices-Speech-Recognition-445`.
3. `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
4. Set your own identity: `git config user.name "<your name>"` and `git config user.email "<your GitHub email>"`.
5. **Update your branch** (it is on GitHub but from 7 October, before the dataset change): `git fetch origin`, `git checkout -b dev_shafil origin/dev_shafil`, `git merge origin/main`, then `git push origin dev_shafil`.
6. Work by hand or with any AI assistant; [CONTRIBUTING.md](../CONTRIBUTING.md) has the steps and a ready-made prompt, and the card for each of your tasks is in **Task cards** at the end of this page. Before running a model, get the audio and build the processed copies once: [docs/RUNS.md](../RUNS.md), steps 1–2. With an assistant that works in the repository (Claude Code, Codex, Cursor, …), just say: *"I'm Shafil (M4). What is completed? Then complete my next tasks."*

## My tasks

| ID | Task | Depends on | Status | Done · date |
|---|---|---|---|---|
| T35 | Visualisation module + tests + example figures ([card](#t35)) | — | 🟡 code + tests done; 👤 example figures from my own recordings | |
| T36 | Demo skeleton `app.py` ([card](#t36)) | T35 | ✅ | 2026-10-18 |
| T38 | Literature notes ([card](#t38)) | — | ✅ | 2026-10-18 |
| T39 | Planning poster v2 ([card](#t39)) | — | ✅ | 2026-10-18 |
| T14 | SVM → `results/svm/fsc/` ([card](#t14)) | — | ✅ | 2026-10-18 |
| T18 | Untrained wav2vec 2.0 baseline → `results/w2v2_untrained/fsc/` ([card](#t18)) | — | ✅ | 2026-10-18 |
| T46 | Linear probe: `support/embeddings.py` + logistic regression → `results/w2v2_probe/fsc/` ([card](#t46)) | — | ✅ | 2026-10-18 |
| T23 | Demo recognising with your SVM + figures command ([card](#t23)) | T14, T35, T36 | 🟡 code + tests done; 👤 live check with my voice | |
| T27 | Results poster, slides, 👤 demo video ([card](#t27)) | T23 | ⬜ | |

**Start with:** T14 (minutes on a laptop), T35 and T18. Your demo uses your own SVM model, so nothing you do waits for anyone.

## Faculty update

Plain-language summary for the weekly project update, newest week first. Add to it at the end of every session: only finished work, and numbers only from `results/`.

**Week of 2026-10-18**
- **SVM (T14):** I trained and evaluated a support vector machine on the 78 MFCC statistics of TenVoices-FSC. It reaches 30.5% accuracy on the random split and 22.3% on speakers it never heard (5 folds pooled; `results/svm/fsc/summary.json`), close to Random Forest: MFCC models depend strongly on knowing the speaker.
- **Untrained baseline and linear probe (T18, T46):** wav2vec 2.0 with an untrained head scores 2.8% on unseen speakers (chance is 2%; `results/w2v2_untrained/fsc/summary.json`). With the same model frozen and only a logistic regression trained on its features (`support/embeddings.py`), accuracy rises to 60.5% on unseen speakers and 69.5% on the random split (`results/w2v2_probe/fsc/summary.json`). Pre-training alone explains most of the gain; full fine-tuning adds the rest.
- **Visualisation and demo (T35, T36, T23):** `support/visualization.py` draws waveforms, spectrograms, MFCCs, the effect of our pre-processing and the results charts; one command (`python -m support.visualization --results results`) draws the comparison, per-speaker and noise figures of every run in `images/`. The demo app (`app.py`) records or uploads a phrase, shows what the model receives and the top-5 phrases from the saved model, with an optional Whisper transcript.
- **Poster and literature (T39, T38):** the planning poster now shows the team of four, the two datasets and the four-stage pipeline with Whisper; the literature notes cover eight checked papers on self-supervised speech models (`docs/literature/self_supervised_speech.md`).

## Requests for the lead

- **Phrase list for the demo (T36):** `support/phrases.py` `load_phrases()` has no `dataset` argument yet and still reads the old `data/prompts/phrases.csv`. The demo therefore uses `support.dataset.load_labels("fsc")` for the 50 TenVoices-FSC commands. If you add `load_phrases(dataset=...)`, `phrase_table()` / `phrase_texts()` in `app.py` can switch to it in one line.
- **Commands for AGENTS.md:** demo `python app.py [--model outputs/models/svm/fsc/random.joblib]`; clip figures `python -m support.visualization clip.wav --out images/example_overview.png [--preprocessing images/example_preprocessing.png]`; results figures `python -m support.visualization --results results --out images`; probe `python -m support.embeddings --dataset fsc [--layer N]`.
- **`main.py` on Windows:** `tests/test_inference.py::test_main_cli_with_benchmark` fails on a Windows console (cp1252): `main.py` prints characters the console cannot encode (`'charmap' codec can't encode characters`). It passes with `PYTHONIOENCODING=utf-8`. A `sys.stdout.reconfigure(encoding="utf-8")` at the start of `main()` would fix it. All my tests pass.

## Handoff log (newest first)

**2026-10-18 · Shafil**
- **Ready to merge:** T14, T18, T36, T38, T39, T46. **Code done, waiting for me (👤):** T35 example figures from my own recordings; T23 live check of the demo with my voice.
- **Files:** `support/visualization.py`, `support/embeddings.py`, `app.py`, `tests/test_visualization.py`, `tests/test_app.py`, `tests/test_embeddings.py`, `tests/_embeddings_check.py`, `docs/literature/self_supervised_speech.md`, `others/poster.html`, `images/poster.png`, `results/svm/fsc/`, `results/w2v2_untrained/fsc/`, `results/w2v2_probe/fsc/`, `images/confusion/{svm,w2v2_untrained,w2v2_probe}_fsc_*.png`, `images/results_*.png`, this page.
- **Key results (TenVoices-FSC, unseen speakers / random split):** SVM 0.223 / 0.305; untrained wav2vec 2.0 0.028 / 0.025; linear probe 0.605 / 0.695.
- **Tests:** 179 run, all pass except `test_inference.test_main_cli_with_benchmark` on a Windows console (see *Requests for the lead*; passes with `PYTHONIOENCODING=utf-8`).
- **Next:** record 2–3 commands for the T35 example figures; try the demo with my voice (T23); then T27 (results poster, slides, demo video). The SVM is weak (22% on unseen speakers), so for the video the demo should load a fine-tuned wav2vec 2.0 model from the Drive (`python app.py --model <folder>`).
- **Blockers:** none.

## Task cards

Written by Tanvir: exactly what to build, which files, which functions to use, the tests and when the task is done.

<a id="t35"></a>
### T35 · Visualisation module (M4 · Shafil)

| | |
|---|---|
| **Goal** | Figures that show what the models "see" (waveform, spectrogram, MFCC) and reusable chart helpers for the results, the demo and the report |
| **Depends on** | T01 ✅ |
| **You create** | `support/visualization.py`, `tests/test_visualization.py`, example figures `images/example_*.png` |
| **Use, don't edit** | `support/audio.py`: `read_audio(path)`, `preprocess(samples, sample_rate, audio_cfg=None, pad=True)`, `load_audio(path, audio_cfg=None, pad=True)`, `peak_dbfs(samples)`; `support/config.py`: `load_config()`, `resolve_path(cfg, key)`; `librosa` (`stft`, `amplitude_to_db`, `feature.melspectrogram`, `feature.mfcc`, `display.specshow`); `matplotlib` |
| **Give your assistant** | `AGENTS.md`, this card, `support/audio.py`, `support/metrics.py` (style example: it draws with `matplotlib.figure.Figure`), `configs/config.yaml`, `requirements.txt` |

#### Build

Draw with `matplotlib.figure.Figure` (not `pyplot`), like `plot_confusion_matrix` in `support/metrics.py`, so the code works in scripts, tests and the Gradio app. Each function returns the `Figure`; an optional `path` argument saves it (PNG, 150 dpi) and creates the folder.

**One clip** (mono float audio + sample rate):
1. `plot_waveform(samples, sample_rate, title=None, path=None)`: amplitude over time in seconds.
2. `plot_spectrogram(samples, sample_rate, kind="linear", title=None, path=None)`: log-power spectrogram in dB; `kind="mel"` gives a 64-band log-mel spectrogram.
3. `plot_mfcc(samples, sample_rate, n_mfcc=13, title=None, path=None)`: MFCC heatmap (for display; the models' features come from `support/features.py` in T11).
4. `clip_overview(samples, sample_rate, title=None, path=None)`: one figure with 4 panels: waveform, spectrogram, log-mel, MFCC.
5. `plot_preprocessing(path_to_audio, audio_cfg=None, out_path=None)`: before / after the shared pre-processing (raw waveform at its own rate vs. the processed 16 kHz, trimmed, normalised, padded waveform), with durations and peak dBFS in the titles.

**Result charts** (pandas DataFrames, so they work with any results table later):
6. `plot_model_comparison(table, metric="accuracy", error_cols=None, path=None)`: bar chart, one bar per row of `table["model"]`; optional `(low, high)` columns for error bars.
7. `plot_accuracy_vs_snr(table, path=None)`: one line per `model`, x = `snr_db`, y = `accuracy`.
8. `plot_per_speaker(table, path=None)`: bars of `accuracy` per `speaker_id`, grouped by `model` when that column exists.

**CLI:** `python -m support.visualization clip.wav --out images/example_overview.png [--preprocessing images/example_preprocessing.png]`.

#### Tests (`tests/test_visualization.py`)

Use synthetic audio (a 1 s sine tone with silence around it, like `tests/test_audio.py`) and tiny DataFrames:
- each function returns a `Figure` with the expected number of axes (the overview has 4 panels);
- saving creates a non-empty file in a temporary folder;
- the chart helpers raise `ValueError` when a required column is missing.

#### Example figures

Record yourself saying 2–3 short commands such as "Turn on the lights" (any recorder app, WAV); these clips are only for the figures, not part of the dataset. Run the CLI to make `images/example_<name>_overview.png` and `images/example_preprocessing.png`. **Do not commit the WAV files.**

#### Done when

- `python -m unittest discover -s tests -v` passes.
- The example figures are in `images/` and look right (axes labelled, units in the titles).

#### For the faculty update

"I built the visualisation module. It shows how a spoken phrase looks as a waveform, a spectrogram and MFCCs, and what our pre-processing changes. It also provides the chart functions for the results. Here are figures of my own voice saying our phrases."

---

<a id="t36"></a>
### T36 · Demo app skeleton (M4 · Shafil)

| | |
|---|---|
| **Goal** | The first version of the live demo: speak or upload a phrase, see what the system hears. The model is connected later (T23), so the app must not show fake predictions |
| **Depends on** | T35 (your own; counts once your page marks it ✅) |
| **You create** | `app.py`, `tests/test_app.py` |
| **Use, don't edit** | `support/audio.py`: `preprocess(samples, sample_rate, audio_cfg=None, pad=False)` (accepts int16 microphone arrays), `peak_dbfs`; `support/phrases.py`: `load_phrases()` → list of `{"phrase_id", "category", "noisy_take", "text"}`; your `support/visualization.py`: `clip_overview(...)`; `gradio` 6.29.1 |
| **Give your assistant** | `AGENTS.md`, this card, `support/audio.py`, `support/phrases.py`, your `support/visualization.py`, `requirements.txt` |

#### Build

1. `process_audio(audio)` (plain function, easy to test): `audio` is what `gr.Audio(type="numpy")` returns, a `(sample_rate, samples)` tuple or `None`.
   - `None` or empty audio → a friendly message and no plot.
   - Otherwise: `preprocess(samples, sample_rate, pad=False)`, then return `(processed audio for playback as (16000, samples), clip_overview figure, a short info text with duration and peak dBFS)`.
   - A clip longer than 6 s after trimming makes `preprocess` raise `ValueError`; catch it and show the message.
2. `build_app()` returns a `gr.Blocks` with:
   - `gr.Audio(sources=["microphone", "upload"], type="numpy")` input and an **Analyse** button,
   - outputs: processed audio player, `gr.Plot` with the overview, info text,
   - a **Prediction** box that says *"Model not connected yet: predictions arrive with the trained model (task T23)"*,
   - a table of the phrases (ID and text) from `load_phrases()`. Once T42 is merged, change that one call to `load_phrases(dataset="fsc")`: the demo recognises the 50 spoken commands of TenVoices-FSC.
3. `main()` with `--port` (default 7860) and `--share` (default off); run with `python app.py`.

#### Tests (`tests/test_app.py`)

- `process_audio(None)` returns the message and no figure.
- `process_audio((48000, int16 tone with silence))` returns 16 kHz audio and a figure with 4 axes.
- A 7-second tone returns the "too long" message instead of raising.
- `build_app()` builds without launching a server.

#### Done when

- `python -m unittest discover -s tests -v` passes.
- `python app.py` opens in the browser; recording and uploading both show the processed audio and the figures.

#### For the faculty update

"I built the first version of the demo app. You can speak into the microphone or upload a clip and see exactly what the system receives after pre-processing. The trained model plugs into the prediction panel later."

---

<a id="t38"></a>
### T38 · Literature notes: self-supervised speech models (M4 · Shafil)

| | |
|---|---|
| **Goal** | Background for report section 2 (related work) on the deep-learning side, and for explaining the untrained baseline you own |
| **Depends on** | — |
| **You create** | `docs/literature/self_supervised_speech.md` |
| **Give your assistant** | `AGENTS.md`, this card, the *References* section of `docs/PROJECT_PLAN.md` |

#### What to cover (6–8 papers)

Starting points (open each link and check it before using it):
1. wav2vec 2.0: Baevski et al., 2020, arXiv:2006.11477 (our main model).
2. HuBERT: Hsu et al., 2021, arXiv:2106.07447.
3. WavLM: Chen et al., 2022, arXiv:2110.13900.
4. XLS-R: Babu et al., 2021, arXiv:2111.09296 (multilingual pre-training).
5. SUPERB: Yang et al., 2021, arXiv:2105.01051 (benchmark of self-supervised models on tasks such as keyword spotting).
6. Speech Commands: Warden, 2018, arXiv:1804.03209 (a spoken-command dataset similar in spirit to ours).

Add 1–2 more of your choice, e.g. on accented English or on fine-tuning with little data. Whisper is covered by M1.

#### Format, per paper

```markdown
##### <Short title> (<first author>, <year>)
- **Reference:** authors, title, venue, year. <DOI or arXiv link>
- **What they did:** 2–3 sentences.
- **What we take from it:** 1–2 sentences, linked to TenVoices (why wav2vec 2.0, why an untrained baseline, what accuracy to expect with little data).
```

End with a short **Summary** paragraph (5–6 sentences) that could become the first draft of the related-work paragraph on self-supervised models.

#### Rules

- **Every reference must be real.** Open each link and check the title, authors, year and venue. AI assistants often invent plausible-looking references: never copy one without opening it.
- Write the notes in your own words.

#### Done when

- 6–8 checked papers in the format above, plus the summary.

#### For the faculty update

"I reviewed the self-supervised speech models (wav2vec 2.0, HuBERT, WavLM, XLS-R) and the spoken-command benchmarks. It explains why we fine-tune wav2vec 2.0 and what the untrained baseline shows."

---

<a id="t39"></a>
### T39 · Planning poster v2 (M4 · Shafil)

| | |
|---|---|
| **Goal** | Bring the planning poster up to date: four team members, the two datasets and the four-stage pipeline (now with Whisper) |
| **Depends on** | — |
| **You edit** | `others/poster.html`, `images/poster.png` |
| **Read, don't edit** | `docs/PROJECT_PLAN.md` (*At a Glance*, *Methodology*, *Hypotheses*), `README.md` (the poster is shown at the top) |
| **Give your assistant** | `AGENTS.md`, this card, `others/poster.html`, the *At a Glance*, *Methodology* and *Hypotheses* parts of `docs/PROJECT_PLAN.md` |

#### Change

1. **Team block:** all four members with their roles (from the team table in `AGENTS.md`).
2. **Pipeline:** four stages: traditional ML (MFCC + k-NN / SVM / RF / XGBoost), untrained wav2vec 2.0 baseline, fine-tuned wav2vec 2.0 (3 / 8 / 15 epochs, white noise), and **Whisper zero-shot** (speech-to-text → nearest phrase, word error rate).
3. **Dataset block:** we no longer record our own audio. Replace the 1,600 self-recorded clips with the two subsets from the *Datasets* section of `docs/PROJECT_PLAN.md`: **TenVoices-FSC** (10 speakers × 50 spoken commands × 2 takes, from Fluent Speech Commands) and **TenVoices-L2** (10 non-native speakers × 50 sentences, from L2-ARCTIC), with their licences. Remove "recorded with consent", the phrase categories and the real noisy takes.
4. **Hypotheses:** H1–H5 as in the project plan (H5 is now about non-native speech).
5. Keep it a **planning edition**: no results, no invented numbers. Keep the 1600 × 1200 px layout and the existing style.

#### Render the PNG

The page is 1600 × 1200 CSS pixels, rendered at 2× (3200 × 2400) for print quality. With Chrome or Chromium:

```bash
### macOS
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
  --screenshot="$PWD/images/poster.png" --window-size=1600,1200 \
  --force-device-scale-factor=2 "file://$PWD/others/poster.html"
### Windows (PowerShell): the same flags with "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

Open the PNG and check that nothing is cut off or overlapping.

#### Done when

- `images/poster.png` is 3200 × 2400 and matches `others/poster.html`.
- The README shows the new poster (same file name, nothing to change there).

#### For the faculty update

"I updated the project poster: our team of four, the two datasets we now use (spoken commands and non-native read speech) and the full four-stage pipeline, including the Whisper speech-to-text comparison."

---

<a id="t14"></a>
### T14 · SVM (M4 · Shafil)

| | |
|---|---|
| **Goal** | A support vector machine with an RBF kernel on the 78 MFCC statistics. Its saved model also powers your demo (T23). |
| **Depends on** | — (shared code and the TenVoices-FSC splits are merged) |
| **You create** | `results/svm/fsc/` and `images/confusion/svm_fsc_*.png` (by running the command) |
| **How** | [RUNS.md](../RUNS.md): get the audio, build the processed copies, run, commit |

#### Run

```bash
python -m support.classical --model svm --dataset fsc
```

#### Done when

- `results/svm/fsc/summary.json` has both `random` and `folds` (5 folds pooled).
- The results folder and the two confusion-matrix images are committed and pushed on your branch.
- The faculty update on your page quotes the accuracies from `summary.json`.

#### For the faculty update

"I trained and evaluated a support vector machine on MFCC features, on seen and on unseen speakers (numbers from `results/svm/fsc/summary.json`); it is the model inside our demo."

---

<a id="t18"></a>
### T18 · untrained wav2vec 2.0 baseline (M4 · Shafil)

| | |
|---|---|
| **Goal** | wav2vec 2.0 with a new, untrained classification head, evaluated without training: it should sit near chance (2% for 50 phrases) and shows that later gains come from fine-tuning. |
| **Depends on** | — (shared code and the TenVoices-FSC splits are merged) |
| **You create** | `results/w2v2_untrained/fsc/` and `images/confusion/w2v2_untrained_fsc_*.png` (by running the command) |
| **How** | [RUNS.md](../RUNS.md): get the audio, build the processed copies, run, commit |

#### Run

```bash
python -m support.train_wav2vec2 --dataset fsc --untrained
```

#### Done when

- `results/w2v2_untrained/fsc/summary.json` has both `random` and `folds` (5 folds pooled).
- The results folder and the two confusion-matrix images are committed and pushed on your branch.
- The faculty update on your page quotes the accuracies from `summary.json`.

#### For the faculty update

"I evaluated wav2vec 2.0 without any training, which shows the chance level and proves our gains come from fine-tuning (numbers from `results/w2v2_untrained/fsc/summary.json`)."

---

<a id="t46"></a>
### T46 · Linear probe on frozen wav2vec 2.0 (M4 · Shafil)

| | |
|---|---|
| **Goal** | Measure how much the *pre-trained* wav2vec 2.0 features already know about the phrases, without any fine-tuning: frozen embeddings + logistic regression. It sits between your untrained baseline (T18) and fine-tuning (T15) and answers "where does the gain come from?" (E10) |
| **Depends on** | — |
| **You create** | `support/embeddings.py`, `tests/_embeddings_check.py`, `tests/test_embeddings.py`, `results/w2v2_probe/fsc/` |
| **Use, don't edit** | `support/dataset.py` (`load_manifest`, `read_processed`), `support/audio.py` (`pad_to_duration`), `support/device.py` (`pick_device`), `support/classical.py` (runs the probe: `--model logreg --features w2v2`), `transformers` (`AutoFeatureExtractor`, `Wav2Vec2Model`) |
| **Give your assistant** | `AGENTS.md`, this card, [RUNS.md](../RUNS.md), `support/dataset.py`, `support/classical.py` (`load_cached_features`), `support/train_wav2vec2.py` (style), `tests/_wav2vec2_check.py` and `tests/test_train_wav2vec2.py` (how PyTorch tests run in a child process) |

#### Build `support/embeddings.py`

1. `embed_clips(samples_list, model, extractor, device, layer)` → one vector per clip: the hidden states of `layer` (default: the last) of the **frozen** model (`model.eval()`, `torch.no_grad()`), averaged over time. Clips are the processed 16 kHz clips (no padding needed: average over the real frames).
2. `extract_dataset(dataset, model_name="facebook/wav2vec2-base", layer=None, cfg=None)` → embeds every clip of `load_manifest(dataset)` and saves `outputs/features/<dataset>_w2v2.npz` with exactly these keys:
   - `utt_ids`: `np.array(ids, dtype=str)` (a plain string array, not objects),
   - `features`: float32, shape `(n_clips, 768)`,
   - `settings`: `np.array(["model=…", "layer=…"], dtype=str)`.
3. CLI: `python -m support.embeddings --dataset fsc [--layer 12] [--device mps]`.

#### Run

```bash
python -m support.embeddings --dataset fsc
python -m support.classical --model logreg --dataset fsc --features w2v2 --run w2v2_probe
```

The second command trains logistic regression (C chosen on validation) on the cached embeddings for the random split and all 5 folds, through the shared evaluation.

#### Tests

PyTorch must not share a process with XGBoost on macOS, so follow the existing pattern: `tests/_embeddings_check.py` saves a tiny random `Wav2Vec2Model` (like `save_tiny_model` in `tests/_wav2vec2_check.py`), embeds a few synthetic clips and writes a JSON report; `tests/test_embeddings.py` runs it with `subprocess` and checks: one vector per clip, the right size, the `.npz` keys and string dtype, the same output on a rerun, and that `load_cached_features` reads it.

#### Done when

- Tests pass; `results/w2v2_probe/fsc/summary.json` has `random` and `folds`; results and confusion images committed.

#### For the faculty update

"I built a linear probe: wav2vec 2.0's frozen features plus logistic regression. Compared with the untrained baseline and with full fine-tuning, it shows how much of the accuracy comes from pre-training alone (numbers from `results/w2v2_probe/fsc/summary.json`)."

---

<a id="t23"></a>
### T23 · Demo with your model, and the figures command (M4 · Shafil)

| | |
|---|---|
| **Goal** | A demo that really recognises phrases, and one command that draws every results figure |
| **Depends on** | T14, T35, T36 (your own) |
| **You edit** | `app.py`, `support/visualization.py` and their tests |
| **Use, don't edit** | `support/inference.py` (`load_recognizer(path).predict(samples, rate, k, texts)`, `WhisperRecognizer`), `support/dataset.py` (`load_labels` for phrase texts), `support/evaluate.py` (results layout) |

#### Build

1. **Demo:** `python app.py [--model outputs/models/svm/fsc/random.joblib]` loads your SVM (T14) by default, and also accepts any saved model folder (e.g. a wav2vec 2.0 model from the Drive). The prediction panel shows the top-5 phrases (ID and text) with a confidence bar chart, replacing the "model not connected yet" box. Optionally, a "Whisper transcript" checkbox uses `WhisperRecognizer`.
2. **Figures command:** `python -m support.visualization --results results --out images` draws, for every run it finds (`support.evaluate.list_runs()`; Tanvir's runs are in `results/M1_tanvir/`): model-comparison bars (random split vs unseen speakers, from `summary.json`), per-speaker bars (`folds/per_speaker.csv`), and accuracy vs SNR (every `noise/*.csv` under `results/`, so far `results/M1_tanvir/noise/`) when noise results exist. Confusion matrices are already drawn by the shared evaluation (`images/confusion/`, Tanvir's in `results/M1_tanvir/figures/confusion/`).
3. Tests: `process_audio` with a small SVM model saved in a temporary folder (see `tests/test_inference.py`), and the figures command on fake results made with `support.evaluate.save_run`.

#### Done when

- Tests pass; the demo recognises your own voice with your SVM; the figures command works on the runs merged so far.

#### For the faculty update

"The demo now recognises spoken commands live with our trained model, and one command draws every results figure from the result files."

---

<a id="t27"></a>
### T27 · Results poster, slides and demo video (M4 · Shafil)

| | |
|---|---|
| **Goal** | The final presentation material |
| **Depends on** | T23 (your own) |
| **You edit** | `others/poster.html`, `images/`, `others/` (slides, video) |

#### Build

- **Poster** (results edition) and **slides**, with the figures from the T23 figures command and numbers copied **only** from `results/all_results.csv` or the runs' `summary.json` (cite the file under each table).
- 👤 **1-minute demo video** of the app recognising spoken commands.
- Keep a short checklist in `others/README.md` of every number and figure and the file it comes from, so the final refresh in T28 (after all runs are merged) is quick.

#### Done when

- Poster, slides and video exist with the results merged so far.

#### For the faculty update

"I made the results poster, the slides and the demo video."

