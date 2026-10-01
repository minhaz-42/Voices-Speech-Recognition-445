# TenVoices: Data Collection Protocol

> **Version 0 (draft).** Pilot-tested with the team's own recordings and frozen as v1 in task T05. Owner: **M2**; reviewed by **M1**.
> Phrase list: [`data/prompts/phrases.csv`](../data/prompts/phrases.csv) · Consent: [consent_form.md](consent_form.md)

## 1. Targets

| | |
|---|---|
| Speakers | 10 (S01–S02 team members, S03–S10 volunteers or members who join later), plus 2 backups; target 5 female / 5 male, self-reported |
| Phrases | 50 classes in 5 categories of 10: commands, numbers, everyday sentences, Bangladeshi names and places, phonetically rich sentences |
| Clips per speaker | 160: 3 clean rounds × 50 phrases + 10 real-noise takes |
| Total | ≈ 1,600 clips, about 1 hour of speech |
| Session length | About 30 minutes, including a short break |

## 2. Roles in a session

- **Operator:** the member who recruited the speaker. Runs the recorder, watches levels, asks for re-takes, writes the session log.
- **Speaker:** reads the phrases.
- **Cross-checker:** a different member who later listens to every clip (pairs in [PROJECT_PLAN.md](PROJECT_PLAN.md), section *Team, Roles and File Ownership*).

## 3. Who can take part

- Adults (18+) who can read English sentences comfortably, recruited from friends, family and classmates.
- Aim for variety: home division (Dhaka, Chattogram, Rajshahi, Khulna, Barishal, Sylhet, Rangpur, Mymensingh), age, first language or dialect, recording device.
- Every speaker signs the consent form **before** recording. Public release is a separate opt-in.

**Recruitment message (template):**
> Hi! For our university speech-recognition project we need 10 volunteers to read 50 short English phrases aloud three times (around 30 minutes). Recordings are anonymous (only a code like S07 is stored). With your permission we will share them openly for research. Would you help? We can record whenever suits you.

## 4. Equipment and settings

- **Recorder:** `tools/recorder.py` (task T02). Use the same tool for every session.
- **Microphone:** the laptop's built-in mic or a wired headset/earphone mic. Use one device per speaker and record it in `speakers.csv`. Different devices across speakers are fine (realistic variety).
- **Format:** WAV, 48 kHz (44.1 kHz is fine), 16-bit, mono. A processed 16 kHz copy is made later.
- **Levels:** peaks between −12 and −3 dBFS, never clipping; mouth 20–30 cm from the mic.
- **Clean condition:** quiet indoor room, door closed, fan/AC off if possible, no other people talking.
- **Noisy condition:** a real noisy place (cafeteria, corridor, roadside, or fan + TV). Note the noise type in metadata.

## 5. Session script

| Step | What happens | Time |
|---|---|---|
| 0 | **Before:** charge the laptop, test the recorder, bring the consent form | — |
| 1 | Welcome, explain the project, sign consent, fill in speaker metadata | 4 min |
| 2 | Mic check + 3 warm-up phrases (not saved) | 2 min |
| 3 | **Round 1:** all 50 phrases in shuffled order | 6 min |
| 4 | **Round 2:** all 50 phrases in a new shuffled order | 6 min |
| 5 | Break: water, stretch | 2 min |
| 6 | **Round 3:** all 50 phrases in a new shuffled order | 6 min |
| 7 | Move to a noisy spot. **Noisy block:** the 10 phrases marked `noisy_take=yes` | 2 min |
| 8 | Thank the speaker, confirm their release choice, back up all files to the shared Drive **the same day** | 2 min |

Three rounds in different orders give natural variation between the repetitions, instead of three identical takes in a row.

## 6. Instructions to read to the speaker

1. Speak naturally, at your normal speed and volume. Don't over-pronounce.
2. Wait until the recorder shows it is recording, pause for about half a second, read, pause again, then stop.
3. Numbers are written as words. Read them exactly as written.
4. If you stumble, cough or read a different word, just press **Re-record**.

## 7. Operator checklist for every take

- [ ] The phrase is read exactly as written. A different word means re-record, because the label must be correct.
- [ ] Nothing cut at the start or end
- [ ] No clipping, and the level meter is in the green
- [ ] Less than 2 s of silence before or after
- [ ] No background voices in clean takes

## 8. File naming and folders

```
data/raw/<SPK>/<SPK>_<PHRASE>_r<REP>_<COND>.wav
e.g.  data/raw/S07/S07_CMD02_r1_clean.wav
      data/raw/S07/S07_CMD02_r3_clean.wav
      data/raw/S07/S07_NER06_r1_noisy.wav
```

- The `utt_id` is the file name without `.wav`.
- The **label** is the phrase ID (e.g. `CMD02`).
- Audio is **never** committed to git: `data/raw/` and `data/processed/` are gitignored. The working copy lives in the private shared Drive folder `TenVoices/raw/<SPK>/`.

## 9. Metadata (CSV schemas)

Templates are in [`data/metadata/`](../data/metadata/).

| File | Columns |
|---|---|
| `phrases.csv` | `phrase_id, category, noisy_take, text` |
| `speakers.csv` | `speaker_id, gender, age_band, home_division, first_language, other_languages, english_use, device, location_type, session_date, operator, consent_signed, consent_release, notes` |
| `utterances.csv` | `utt_id, speaker_id, phrase_id, category, rep, condition, noise_type, text, duration_s, orig_sample_rate, peak_dbfs, qc_status, qc_by, notes` |

- **Value sets:**
  - `gender`: self-described (female / male / other / prefer not to say).
  - `age_band`: 18–24, 25–34, 35–49, 50+.
  - `english_use`: daily / weekly / rarely.
  - `condition`: clean / noisy.
  - `rep`: 1–3 for clean takes, 1 for noisy takes.
  - `qc_status`: ok / rerecord / drop.

## 10. Label rules

- Each clip's label is the phrase ID of what was read; there are no free transcripts.
- If the speaker reads a different word, adds words or stops halfway, the take is **re-recorded, never relabelled**.
- Phrase text uses standard spellings: Mohakhali, Sylhet, Chattogram, Cox's Bazar, Dhanmondi, Sreemangal, Sirajganj.

## 11. Quality control

**Automatic** (`tools/qc_report.py`, M2):

| Check | Rule |
|---|---|
| Completeness | Every planned (speaker, phrase, rep, condition) file exists |
| Duration | 0.5–8 s |
| Clipping | Peak below −0.5 dBFS |
| Silence | Leading and trailing silence under 2 s each |
| Noise floor | Estimated SNR ≥ 15 dB for clean takes |
| Format | Readable WAV, mono, expected sample rate |

**Human:**
- The operator listens during the session.
- The cross-checker listens to **100%** of the speaker's clips and checks that each one matches its phrase.
- Every clip ends with a `qc_status`. The target is ≥ 97% `ok` after re-records (task T07).

## 12. Privacy and storage

- Speakers are identified only by pseudonymous IDs (S01–S10). Names never appear in file names, metadata or git.
- Signed consent forms (scans/photos) are kept in a separate restricted Drive folder, never in the repository.
- If a speaker withdraws before release, delete all their files and note the deletion in the session log.
- Only speakers who opted in to public release are included in the published dataset.

## 13. Release package (task T26)

`tenvoices-v1.0.zip`, published as a GitHub Release asset. It contains:
- FLAC 16 kHz mono audio (consenting speakers only)
- `speakers.csv`, `utterances.csv`, `phrases.csv`
- `data/splits/`
- the datasheet
- the CC BY 4.0 licence
- a SHA-256 manifest

Phrase sources are credited in the datasheet: Harvard sentences (IEEE 1969 Recommended Practice) and phrases written by the team.
