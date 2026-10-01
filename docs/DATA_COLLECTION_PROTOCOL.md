# TenVoices: Data Collection Protocol

> **Version 0 (draft).** Pilot-tested with the team's own recordings and frozen as v1 in **Week 1**. Owner: **M2** (dataset); reviewed by **M1**.
> Prompts: [`data/prompts/prompts_v0.csv`](../data/prompts/prompts_v0.csv) · Consent: [consent_form.md](consent_form.md)

## 1. Targets

| | |
|---|---|
| Speakers | 10 (S01–S02 team members, S03–S10 volunteers or members who join later), plus 2 backups; target 5 female / 5 male, self-reported |
| Recordings per speaker | 120 (75 shared-prompt takes, 30 unique sentences, 5 spontaneous answers, 10 real-noise takes) |
| Total | ≈ 1,200 utterances, ≈ 70 minutes of speech |
| Session length | 30–35 minutes including a short break |

## 2. Roles in a session

- **Operator:** the member who recruited the speaker. Runs the recorder, watches levels, asks for re-takes, writes the session log.
- **Speaker:** reads the prompts.
- **Cross-checker:** a different member who later listens to every file and verifies the transcripts (pairs in [PROJECT_PLAN.md](PROJECT_PLAN.md), section *Team, Roles and File Ownership*).

## 3. Who can take part

- Adults (18+) who can read English sentences comfortably, recruited from friends, family and classmates.
- Aim for variety: home division (Dhaka, Chattogram, Rajshahi, Khulna, Barishal, Sylhet, Rangpur, Mymensingh), age, first language or dialect, recording device.
- Every speaker signs the consent form **before** recording. Public release is a separate opt-in.

**Recruitment message (template):**
> Hi! For our university speech-recognition project we need 10 volunteers to read about 100 short English sentences aloud (around 30 minutes). Recordings are anonymous (only a code like S07 is stored). With your permission we will share them openly for research. Would you help? We can record whenever suits you.

## 4. Equipment and settings

- **Recorder:** `tools/recorder.py` (built in Week 1). Use the same tool for every session.
- **Microphone:** the laptop's built-in mic or a wired headset/earphone mic. Use one device per speaker and record it in `speakers.csv`. Different devices across speakers are fine (realistic variety).
- **Format:** WAV, 48 kHz (44.1 kHz is fine), 16-bit, mono. A processed 16 kHz copy is made later.
- **Levels:** peaks between −12 and −3 dBFS, never clipping; mouth 20–30 cm from the mic.
- **Clean condition:** quiet indoor room, door closed, fan/AC off if possible, no other people talking.
- **Noisy condition:** a real noisy place (cafeteria, corridor, roadside, or fan + TV). Note the noise type in metadata.

## 5. Session script

| Step | What happens | Time |
|---|---|---|
| 0 | **Before:** charge the laptop, test the recorder, prepare this speaker's prompt sheet (shared + assigned unique sentences), bring the consent form | — |
| 1 | Welcome, explain the project, sign consent, fill in speaker metadata | 4 min |
| 2 | Mic check + 3 warm-up prompts (not saved) | 2 min |
| 3 | **Block 1:** commands, repetition 1 (CMD × 15) | 3 min |
| 4 | **Block 2:** numbers (NUM × 10) | 2 min |
| 5 | **Block 3:** phonetic sentences (PHO × 15) | 3 min |
| 6 | **Block 4:** everyday sentences and questions (EVQ × 10) | 2 min |
| 7 | **Block 5:** Bangladeshi names and places (NER × 10) | 2 min |
| 8 | Break: water, stretch | 2 min |
| 9 | **Block 6:** unique sentences (UNQ × 30) | 6 min |
| 10 | **Block 7:** commands, repetition 2 (CMD × 15, shuffled order) | 3 min |
| 11 | **Block 8:** spontaneous answers (SPN × 5, 10–20 s each) | 3 min |
| 12 | Move to a noisy spot. **Block 9:** the 10 prompts flagged `noisy_rerecord=yes` | 3 min |
| 13 | Thank the speaker, confirm their release choice, back up all files to the shared Drive **the same day** | 2 min |

Commands are recorded at the start and again near the end. Separating the two repetitions in time makes the speaker-dependent DTW test realistic.

## 6. Instructions to read to the speaker

1. Speak naturally, at your normal speed and volume. Don't over-pronounce.
2. Wait until the recorder shows it is recording, pause for about half a second, read, pause again, then stop.
3. Numbers are written as words. Read them exactly as written.
4. If you stumble or cough, just press **Re-record**.
5. For the open questions, answer in 1–3 sentences. **Don't mention your name, address, phone number or anything personal.**

## 7. Operator checklist for every take

- [ ] The whole sentence is spoken, with nothing cut at the start or end
- [ ] No clipping, and the level meter is in the green
- [ ] Less than 2 s of silence before or after
- [ ] No background voices in clean takes
- [ ] The words match the prompt. A small natural deviation is fine: keep the take and note it so the transcript is corrected to what was actually said.

## 8. File naming and folders

```
data/raw/<SPK>/<SPK>_<PROMPT>_r<REP>_<COND>.wav
e.g.  data/raw/S07/S07_CMD05_r2_clean.wav
      data/raw/S07/S07_UNQ143_r1_clean.wav
      data/raw/S07/S07_NER06_r1_noisy.wav
      data/raw/S07/S07_SPN03_r1_clean.wav
```

The `utt_id` is the file name without `.wav`. Audio is **never** committed to git: `data/raw/` and `data/processed/` are gitignored, and the working copy lives in the private shared Drive folder `TenVoices/raw/<SPK>/`.

## 9. Metadata (CSV schemas)

Templates are in [`data/metadata/`](../data/metadata/).

| File | Columns |
|---|---|
| `prompts.csv` | `prompt_id, category, text_group, repetitions, noisy_rerecord, text` |
| `speakers.csv` | `speaker_id, gender, age_band, home_division, first_language, other_languages, english_use, device, location_type, session_date, operator, consent_signed, consent_release, notes` |
| `utterances.csv` | `utt_id, speaker_id, prompt_id, category, text_group, rep, condition, noise_type, prompt_text, transcript, duration_s, orig_sample_rate, peak_dbfs, qc_status, qc_by, notes` |

- **Value sets:**
  - `gender`: self-described (female / male / other / prefer not to say).
  - `age_band`: 18–24, 25–34, 35–49, 50+.
  - `english_use`: daily / weekly / rarely.
  - `condition`: clean / noisy.
  - `qc_status`: ok / fixed_transcript / rerecord / drop.
- **Text groups:**
  - `A`: may be used for training.
  - `B`: held-out text, never trained on.
  - `U`: unique sentence, read by one speaker only.
  - `S`: spontaneous answer, test only.

## 10. Transcription conventions (verbatim)

- The transcript is **what was actually said**. If it matches the prompt, copy the prompt.
- Numbers are written as words, both in prompts and in spontaneous transcripts.
- Leave out filled pauses (*uh, um*) and cut-off partial words. Keep real repetitions and self-corrections as spoken.
- Use standard spellings for names: Mohakhali, Sylhet, Chattogram, Cox's Bazar, Dhanmondi, Sreemangal, Sirajganj.
- Mark unintelligible words as `[unk]`. Utterances containing `[unk]` are left out of scoring and counted in the datasheet.
- Scoring applies the same normaliser to references and hypotheses (lower case, no punctuation, numbers and contractions normalised; see `docs/METRICS_SPEC.md`, written in Week 1).

## 11. Quality control

**Automatic** (`tools/qc_report.py`, M2):

| Check | Rule |
|---|---|
| Completeness | Every planned (speaker, prompt, rep, condition) file exists |
| Duration | 0.6–20 s (spontaneous: up to 25 s) |
| Clipping | Peak below −0.5 dBFS |
| Silence | Leading and trailing silence under 2 s each |
| Noise floor | Estimated SNR ≥ 15 dB for clean takes |
| Format | Readable WAV, mono, expected sample rate |

**Human:**
- The operator spot-listens during the session.
- The cross-checker listens to **100%** of the speaker's files and verifies every transcript.
- Every take ends with a `qc_status`. The target is ≥ 97% `ok` or `fixed_transcript` after re-records in Week 2.

## 12. Noise bank (for augmentation and noise tests)

- M2 records 20 ambient clips of 60 s each across 5 types: fan/AC, traffic, cafeteria babble, rain, household. No clear speech.
- Clips are split by file into **train-noise (12)** and **test-noise (8)**. The two sets never mix: training augmentation uses only train-noise, and the noise sweep uses only test-noise (plus generated white noise).

## 13. Privacy and storage

- Speakers are identified only by pseudonymous IDs (S01–S10). Names never appear in file names, metadata or git.
- Signed consent forms (scans/photos) are kept in a separate restricted Drive folder, never in the repository.
- If a speaker withdraws before release, delete all their files and note the deletion in the session log.
- Only speakers who opted in to public release are included in the published dataset.

## 14. Release package (Week 6)

`tenvoices-v1.0.zip`, published as a GitHub Release asset. It contains:
- FLAC 16 kHz mono audio (consenting speakers only)
- `speakers.csv`, `utterances.csv`, `prompts.csv`
- `data/splits/folds.json`
- the datasheet
- the CC BY 4.0 licence
- a SHA-256 manifest

Prompt sources are credited in the datasheet:
- Harvard sentences (IEEE 1969 Recommended Practice)
- Mozilla Common Voice sentence corpus (CC0)
- sentences written by the team
