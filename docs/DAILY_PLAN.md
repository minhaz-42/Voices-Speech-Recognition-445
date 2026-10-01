# TenVoices: Day-by-Day Plan (42 days)

> **Sat 3 Oct → Fri 13 Nov 2026.** Read [PROJECT_PLAN.md](PROJECT_PLAN.md) first.
> If the instructor's real dates differ, shift the dates; the order of the work stays valid.

**Members:** **M1** Shakil Ahmed (data & pipeline) · **M2** Fahim Foysal (ASR models) · **M3** Shefa Tabassum (collection, augmentation & demo) · **M4** Tanvir Ahmed (evaluation & experiments)

## How we use this plan

- **Weekly rhythm:**
  - **Saturday:** 30-min meeting, merge last week's PRs, check exit criteria.
  - **Sunday–Thursday:** tasks.
  - **Friday:** buffer (catch up, no new tasks).
- **Daily stand-up** (group chat, 10 pm):

  ```
  [Day N · Name]  Done: …  |  Next: …  |  Blocked: none / …
  ```

- **Daily log:** add 2–3 lines to `docs/logs/<you>.md`. It becomes your weekly log in the update report.
- **Unfinished tasks** move to Friday. If a task is blocked for more than a day, call a 10-minute meeting.
- **Ticking tasks:** only **M1** edits this file, at the Saturday meeting. Day-to-day progress is ticked in your weekly GitHub issue.

### Recording and cross-check schedule

| Speaker | Recorded on | Operator | Cross-checked by |
|---|---|---|---|
| S01–S04 (team) | Day 5 · Wed 7 Oct (pilot) | self | S01 → M2 · S02 → M1 · S03 → M4 · S04 → M3 (Day 8) |
| S05 | Day 8 · Sat 10 Oct | M1 | M2 (Day 9) |
| S06 | Day 9 · Sun 11 Oct | M1 | M2 (Day 10) |
| S07 | Day 9 · Sun 11 Oct | M2 | M1 (Day 10) |
| S08 | Day 10 · Mon 12 Oct | M3 | M4 (Day 11) |
| S09 | Day 11 · Tue 13 Oct | M3 | M4 (Day 12) |
| S10 | Day 11 · Tue 13 Oct | M4 | M3 (Day 12) |
| Backups / re-records | Day 12 · Wed 14 Oct | all | |

---

## Week 1: Foundations and pilot (Sat 3 – Fri 9 Oct) → Milestone M1 "Ready to record"

### Day 1 · Sat 3 Oct · Kick-off
- [ ] **All:** 60-min kick-off. Read PROJECT_PLAN together, settle the 5 decisions in §16, fill in the README course table, fix the stand-up time.
- [ ] **Tanvir** (holds this prepared folder): push it to a new **public** GitHub repo and add the other three as collaborators.
- [ ] **M1:** protect `main`, create the branches `member1-data`, `member2-asr`, `member3-collection`, `member4-evaluation`, plus labels and a Project board with the Week 1 issues.
- [ ] **M2:** set up a Python 3.11 env (local + Colab), install `requirements.txt` and pin the versions that work. Transcribe one test clip with Whisper-tiny and note the speed.
- [ ] **M3:** finalise the consent form and recruitment message. Shortlist volunteers (6 + 2 backups) to reach 5F/5M overall.
- [ ] **M4:** write `docs/METRICS_SPEC.md`: WER/CER/SER definitions, micro vs macro averaging, normalisation rules.
- ✅ **Done when:** the repo is live and everyone can push to their own branch.

### Day 2 · Sun 4 Oct
- [ ] **M1:** build the unique-sentence pool. Download the Common Voice English sentences (CC0; fallback: Harvard lists 3–32), keep 6–14-word sentences, remove duplicates and odd ones, then seed-sample 300 → `data/prompts/unique_pool.csv` (30 per speaker).
- [ ] **M2:** `support/asr.py`: the `ASREngine` interface + a Whisper engine (Hugging Face `transformers`).
- [ ] **M3:** `tools/recorder.py` v0 (Gradio): shows the prompt, records, saves a 48 kHz WAV with the naming convention.
- [ ] **M4:** `support/metrics.py`: WER/CER/SER with `jiwer` + normaliser, with unit tests on hand-computed examples.
- [ ] **All:** review `prompts_v0.csv` in a PR and comment by tonight.
- ✅ **Done when:** prompts v1 (60 shared + 300-sentence pool) is merged.

### Day 3 · Mon 5 Oct
- [ ] **M1:** `configs/config.yaml` + `support/config.py` (paths, sample rate, seeds, model names, validation).
- [ ] **M2:** wav2vec2-CTC and Vosk engines behind the same interface.
- [ ] **M3:** recorder v1: re-record button, progress bar, block order, writes a row to `utterances.csv`.
- [ ] **M4:** transcription + normalisation conventions with tests ("six thirty" vs "6:30", "it's" vs "it is", "Cox's Bazar").
- ✅ **Done when:** one clip goes record → save → transcribe → score on one laptop.

### Day 4 · Tue 6 Oct
- [ ] **M1:** `support/audio.py`: load, mono, resample to 16 kHz, loudness normalisation, Silero VAD trim.
- [ ] **M2:** batch zero-shot runner: a folder of audio → `outputs/hyp_<model>.csv`, with timing for RTF.
- [ ] **M3:** printable per-speaker prompt sheets, session checklist, list of noisy locations.
- [ ] **M4:** `support/features.py` (MFCC + Δ + ΔΔ, CMVN) and a DTW prototype (`librosa.sequence.dtw`).
- ✅ **Done when:** every member's smoke test runs from their own branch.

### Day 5 · Wed 7 Oct · Pilot day
- [ ] **All:** record your own full session with the recorder (~35 min each): S01 Shakil, S02 Fahim, S03 Shefa, S04 Tanvir. Back up to the shared Drive the same day.
- [ ] **M1:** `tools/qc_report.py` v0 (duration, clipping, silence, SNR estimate); run it on the pilot.
- [ ] **M2:** Whisper-base zero-shot on the pilot audio.
- [ ] **M3:** collect usability problems and fix recorder bugs.
- [ ] **M4:** first pilot WER/CER; inspect 20 errors for normalisation issues.
- ✅ **Done when:** 4 × 120 pilot recordings are saved and backed up, and the first WER exists.

### Day 6 · Thu 8 Oct · Milestone M1
- [ ] **M1:** `support/dataset.py` (reads metadata, filters subsets), a draft `tools/make_splits.py`, and a leakage unit test.
- [ ] **M2:** Colab notebook skeleton for Whisper LoRA fine-tuning (PEFT); dry run on 20 pilot clips.
- [ ] **M3:** apply the pilot fixes, **freeze protocol v1**, book volunteer sessions for Days 8–11, send consent forms ahead of time.
- [ ] **M4:** results-CSV schema + experiment config format; speaker-dependent DTW on the S01–S04 commands (rep 1 vs rep 2).
- ✅ **Done when:** **M1 Ready to record**: protocol frozen; recorder, QC and metrics working.

### Day 7 · Fri 9 Oct · Buffer
- [ ] Catch up, re-record bad pilot takes, finish the week-1 log.

---

## Week 2: Data collection and QC (Sat 10 – Fri 16 Oct) → Milestone M2 "TenVoices v1 frozen"

### Day 8 · Sat 10 Oct
- [ ] **All:** weekly meeting; merge the Week 1 PRs → tag `v0.1-tools`.
- [ ] **M1:** run session **S05** and QC it. Cross-check S02.
- [ ] **M2:** zero-shot Whisper tiny/base/small on S01–S04 and log hallucinations. Cross-check S01.
- [ ] **M3:** support the S05 session (second operator, session log). Cross-check S04.
- [ ] **M4:** cross-check S03; update the speaker-dependent DTW results.
- ✅ **Done when:** S05 is recorded and the pilot transcripts are verified.

### Day 9 · Sun 11 Oct
- [ ] **M1:** run session **S06** and QC it.
- [ ] **M2:** run session **S07** and QC it. Cross-check S05.
- [ ] **M3:** noise bank part 1: fan/AC, traffic, household (10 × 60 s, no speech).
- [ ] **M4:** per-speaker / per-category scoring functions in `support/experiments.py`.
- ✅ **Done when:** 7 / 10 speakers are recorded.

### Day 10 · Mon 12 Oct
- [ ] **M1:** cross-check S07; QC summary → `outputs/qc_summary.csv` (counts, durations, flags per speaker).
- [ ] **M2:** cross-check S06; wav2vec2 + Vosk zero-shot on S01–S07.
- [ ] **M3:** run session **S08** and QC it.
- [ ] **M4:** speaker-independent DTW prototype on the S01–S07 commands.
- ✅ **Done when:** 8 / 10 speakers are recorded.

### Day 11 · Tue 13 Oct
- [ ] **M1:** compile the re-record list from QC flags and cross-check notes.
- [ ] **M2:** smoke-run the fine-tuning notebook on S01–S07 (a pipeline check, not an experiment).
- [ ] **M3:** run session **S09** and QC it.
- [ ] **M4:** run session **S10** and QC it. Cross-check S08.
- ✅ **Done when:** 10 / 10 speakers are recorded.

### Day 12 · Wed 14 Oct
- [ ] **All:** re-record flagged takes (target ≤ 3% of takes). The operator transcribes the spontaneous answers; the cross-checker verifies them.
- [ ] **M1:** final QC run; `tools/make_splits.py` → `data/splits/folds.json` (5 folds, gender-balanced, one dev speaker per fold).
- [ ] **M2:** help with re-records; finish the spontaneous-transcript checks for S01, S05, S06.
- [ ] **M3:** cross-check S10; dataset statistics figure → `images/dataset_stats.png`.
- [ ] **M4:** cross-check S09; unit tests proving there is no speaker or text leakage in any fold.
- ✅ **Done when:** ≥ 97% of takes pass QC.

### Day 13 · Thu 15 Oct · Milestone M2
- [ ] **M1:** process all audio (16 kHz mono, normalised; VAD-trimmed copies) + a manifest with SHA-256 checksums; upload to the shared Drive; tag `data-v1`.
- [ ] **M2:** freeze the zero-shot model list and decoding settings (beam 5, temperature 0, language "en").
- [ ] **M3:** datasheet `data/README.md`: motivation, composition, collection, preprocessing, uses, distribution.
- [ ] **M4:** `docs/ANALYSIS_PLAN.md`: hypotheses H1–H4, metrics, subsets, statistical tests. **Committed before any main experiment.**
- ✅ **Done when:** **M2 TenVoices v1 frozen.** Any later data change needs a version bump.

### Day 14 · Fri 16 Oct · Buffer
- [ ] Catch up; finish the week-2 log.

---

## Week 3: Baselines and evaluation harness (Sat 17 – Fri 23 Oct) → Milestone M3 "Baselines + Update 1"

### Day 15 · Sat 17 Oct
- [ ] **All:** weekly meeting; merge → tag `v0.2-data`.
- [ ] **M1:** connect `dataset.py` to the frozen manifest; add `main.py transcribe --audio <file> --model <name>`.
- [ ] **M2:** full zero-shot runs on all 1,200 clips: Whisper tiny/base/small, wav2vec2, Vosk (Colab).
- [ ] **M3:** `support/visualization.py`: WER bar chart with CIs, per-speaker heatmap.
- [ ] **M4:** `support/experiments.py`: reads hypothesis CSVs and scores per fold / speaker / category / condition.
- ✅ **Done when:** all zero-shot hypothesis CSVs exist.

### Day 16 · Sun 18 Oct
- [ ] **M1:** unit tests for `audio.py` and `dataset.py`; a GitHub Actions workflow that runs the tests (no model downloads).
- [ ] **M2:** *(Could)* Whisper large-v3-turbo upper-bound run on Colab.
- [ ] **M3:** command confusion-matrix plot; example-transcripts figure.
- [ ] **M4:** final DTW runs, speaker-dependent and speaker-independent → command accuracy table (E1).
- ✅ **Done when:** a draft "zero-shot baselines" table is shared.

### Day 17 · Mon 19 Oct
- [ ] **M1:** RTF / latency benchmark script (with M2).
- [ ] **M2:** decoding sweep on **dev speakers only** (beam size, temperature fallback).
- [ ] **M3:** noise bank part 2: cafeteria babble, crowd, rain (10 × 60 s). Split all noise clips into *train-noise* and *test-noise*.
- [ ] **M4:** `support/stats.py`: paired bootstrap CI + Wilcoxon, with tests.
- ✅ **Done when:** the stats module has been tested on the baseline outputs.

### Day 18 · Tue 20 Oct
- [ ] **M1:** README dataset summary + usage section (only commands that actually work).
- [ ] **M2:** baseline error analysis: named-entity failures, top substitutions, hallucinations.
- [ ] **M3:** `support/augment.py`: noise at a target SNR (train-noise + white), speed 0.9–1.1, gain ±6 dB.
- [ ] **M4:** baseline tables (WER/CER by model × subset, per speaker, per category) → `outputs/tables/`.
- [ ] **All:** 20-min call to review the baseline results.
- ✅ **Done when:** the E2 results are agreed.

### Day 19 · Wed 21 Oct
- [ ] **All:** write your individual update report ([template](templates/individual_update_report.md)) from your daily log.
- [ ] **M2:** domain lexicon v1 `configs/lexicon_bd.txt` from public lists (64 districts, Dhaka areas, universities, common names). Frozen before any test decoding uses it.
- [ ] **M4:** share the baseline and DTW figures so everyone can use them in their reports.
- ✅ **Done when:** report drafts are shared for a quick peer read.

### Day 20 · Thu 22 Oct · Milestone M3 + Update 1
- [ ] **All:** finalise and submit the update reports (move this to the instructor's real date); push code; update the README progress section.
- ✅ **Done when:** **M3 Baselines + Update 1.**

### Day 21 · Fri 23 Oct · Buffer

---

## Week 4: Our method (Sat 24 – Fri 30 Oct) → Milestone M4 "Method results"

### Day 22 · Sat 24 Oct
- [ ] **All:** weekly meeting; merge → tag `v0.3-baselines`.
- [ ] **M1:** VAD front-end in the inference path (`--vad` flag); measure its effect on hallucinations.
- [ ] **M2:** domain-aware decoding (lexicon prompt) for zero-shot Whisper on all folds.
- [ ] **M3:** plug on-the-fly augmentation into the training data pipeline; listen to 20 augmented samples.
- [ ] **M4:** `support/commands.py`: fuzzy matcher + rejection threshold tuned on dev speakers.
- ✅ **Done when:** "zero-shot + prompt" and "zero-shot + VAD" results exist.

### Day 23 · Sun 25 Oct
- [ ] **M1:** per-fold train/dev/test manifests for fine-tuning.
- [ ] **M2:** LoRA fine-tune **fold 1** (Whisper-small, early stopping on the dev speaker).
- [ ] **M3:** run **fold 2** on your Colab with the same notebook and config.
- [ ] **M4:** score fold 1 against zero-shot; sanity-check the learning curves.
- ✅ **Done when:** fold 1 improves dev WER, or the problem has been raised in chat.

### Day 24 · Mon 26 Oct
- [ ] **M1:** run **fold 5** on your Colab.
- [ ] **M2:** run **folds 3–4**.
- [ ] **M3:** Gradio demo `app.py` v0 (microphone → transcript, model switch).
- [ ] **M4:** score folds as they finish; compare them per fold.
- ✅ **Done when:** all 5 folds are trained.

### Day 25 · Tue 27 Oct
- [ ] **M1:** adapter storage (shared Drive / release assets) + `tools/fetch_adapters.py`.
- [ ] **M2:** Whisper-base + LoRA on all 5 folds (model-size comparison).
- [ ] **M3:** demo command mode + example buttons.
- [ ] **M4:** main results table: zero-shot → +VAD → +prompt → +LoRA → full pipeline, with a paired bootstrap against the best baseline (H1).
- ✅ **Done when:** the main table draft exists.

### Day 26 · Wed 28 Oct
- [ ] **M1:** `main.py evaluate --config configs/exp_main.yaml` end to end; tests for the inference path.
- [ ] **M2:** ablations: full pipeline minus {VAD, prompt, LoRA, augmentation} (E4).
- [ ] **M3:** figures: main results with CIs; per-speaker improvement plot.
- [ ] **M4:** named-entity accuracy table (H2) and command-task table (H3, E5).
- ✅ **Done when:** E3–E5 numbers exist.

### Day 27 · Thu 29 Oct · Milestone M4
- [ ] **All:** 45-min results review: what do H1–H4 say? Decide what gets fixed and what is reported as is. Then fix what the review found.
- ✅ **Done when:** **M4 Method results for all 5 folds.**

### Day 28 · Fri 30 Oct · Buffer

---

## Week 5: Robustness, error analysis and demo (Sat 31 Oct – Fri 6 Nov) → Milestone M5 "Results frozen"

### Day 29 · Sat 31 Oct
- [ ] **All:** weekly meeting; merge → tag `v0.4-method`.
- [ ] **M1:** `main.py reproduce`: one command regenerates every table and figure from the data.
- [ ] **M2:** real-noise subset results, zero-shot vs ours.
- [ ] **M3:** synthetic noisy test sets (test-noise clips + white noise at 20/10/5/0 dB).
- [ ] **M4:** noise-sweep scoring (E6) → WER-vs-SNR table.
- ✅ **Done when:** E6 numbers exist.

### Day 30 · Sun 1 Nov
- [ ] **M1:** reproducibility check from a fresh environment against the stored outputs.
- [ ] **M2:** efficiency: RTF, latency, model + adapter size on laptop CPU vs GPU (E8).
- [ ] **M3:** demo v1 polish (examples, live microphone, WER shown when a reference is given); WER-vs-SNR figure.
- [ ] **M4:** error analysis (E7): substitution/deletion/insertion breakdown, top 20 confusions, WER vs length, qualitative examples.
- ✅ **Done when:** E7 and E8 are done.

### Day 31 · Mon 2 Nov
- [ ] **M1:** GitHub Actions green; README "How to run" checked command by command.
- [ ] **M2:** *(Could)* Bangla mini-extension or speaker-ID side study if ahead of schedule; otherwise help M4.
- [ ] **M3:** draft the results-poster layout.
- [ ] **M4:** final list of tables and figures for the report; check that every number traces to an `outputs/` CSV.
- ✅ **Done when:** the figure list is agreed.

### Day 32 · Tue 3 Nov
- [ ] **All:** start the final report from the Group 5 IEEE LaTeX template. Each owner drafts their section (PROJECT_PLAN §12).
- ✅ **Done when:** every section has a skeleton in `others/final_report.tex`.

### Day 33 · Wed 4 Nov · Feature freeze
- [ ] **All:** no new experiments after today, only bug fixes. Generate the final figures; keep drafting.
- ✅ **Done when:** the final figures are committed to `images/`.

### Day 34 · Thu 5 Nov · Milestone M5
- [ ] **All:** first complete report draft; tag `v0.9-results`.
- ✅ **Done when:** **M5 Results frozen.**

### Day 35 · Fri 6 Nov · Buffer

---

## Week 6: Report, poster, video and release (Sat 7 – Fri 13 Nov) → Milestone M6 "Final submission"

### Day 36 · Sat 7 Nov
- [ ] **All:** weekly meeting; each member reviews another's report section (M1→M2, M2→M3, M3→M4, M4→M1).
- [ ] **M1:** integrate the sections, compile with `tectonic`, fix references.
- [ ] **M2:** slides draft (12–15) built from the report figures.
- [ ] **M3:** results poster → `images/poster.png` (+ print PDF), replacing the planning poster.
- [ ] **M4:** polish the results section and threats to validity.

### Day 37 · Sun 8 Nov
- [ ] **M1:** dataset release package (consenting speakers only, FLAC 16 kHz, metadata, datasheet, CC BY 4.0, checksums) → GitHub Release asset.
- [ ] **M2:** 1-minute voice-over script + demo storyboard.
- [ ] **M3:** record the demo video (CLI run → live Gradio recording → results figure) + a GIF for the README.
- [ ] **M4:** final numbers for the abstract and conclusion; check every claim against the tables.

### Day 38 · Mon 9 Nov
- [ ] **All:** clean-clone test. A member who didn't write the code follows the README on a fresh machine or Colab and reproduces one table.
- [ ] **M1:** final README: results, figures, poster, GIF, how to run, dataset link, citation.
- [ ] **M2:** final slides; rehearsal 1.
- [ ] **M3:** final video edit → `others/demo_video.mp4`.
- [ ] **M4:** report proofreading, pass 1.

### Day 39 · Tue 10 Nov
- [ ] **All:** timed rehearsal 2; apply the feedback; proofreading pass 2; check the 8-page limit.

### Day 40 · Wed 11 Nov
- [ ] **All:** work through the final checklist below; tag `v1.0`.

### Day 41 · Thu 12 Nov · Milestone M6: Final submission
- [ ] Submit (move to the instructor's real date).

### Day 42 · Fri 13 Nov · Buffer / presentation prep

---

## Final submission checklist

- [ ] Final report PDF (8 pages, IEEE) + LaTeX source in `others/`
- [ ] Slides + voice-over script in `others/`
- [ ] 1-minute demo video in `others/`, GIF in `images/`
- [ ] Results poster at `images/poster.png`
- [ ] README with results, figures, a working how-to-run and the dataset link
- [ ] Dataset release asset + datasheet + checksums (consenting speakers only)
- [ ] All tests pass on GitHub Actions
- [ ] Every number in the report, README, poster and slides matches `outputs/`
- [ ] Individual reports submitted
- [ ] Git tag `v1.0`
