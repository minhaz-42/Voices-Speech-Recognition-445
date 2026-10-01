# TenVoices: Speech Recognition System on a Self-Collected 10-Speaker Dataset

## Project Information

| Item       | Details                                   |
| ---------- | ----------------------------------------- |
| Course     | *TBD (filled in at kick-off)*             |
| Section    | *TBD*                                     |
| Group      | *TBD*                                     |
| Instructor | *TBD*                                     |
| University | North South University (NSU), Dhaka       |

## Team Members

| Member   | Name           | Planned Contribution                                                                                     | Status  |
| -------- | -------------- | -------------------------------------------------------------------------------------------------------- | ------- |
| Member 1 | Shakil Ahmed   | Repository setup, configuration, audio pre-processing, dataset loader and splits, integration (`main.py`) | Planned |
| Member 2 | Fahim Foysal   | ASR models (Whisper, wav2vec2, Vosk), domain-aware decoding, LoRA fine-tuning                            | Planned |
| Member 3 | Shefa Tabassum | Data collection (recorder tool, protocol, consent), noise augmentation, visualisation, demo app, poster  | Planned |
| Member 4 | Tanvir Ahmed   | Evaluation metrics, classical MFCC + DTW baseline, experiments, statistics, error analysis               | Planned |

<p align="center">
  <img src="images/poster.png" alt="TenVoices project poster" width="850">
</p>

<p align="center">
  <em>Project poster (planning edition). It will be replaced by the results edition at the end of the project.</em>
</p>

## Short Overview

This repository is for a semester project on speech recognition. The project will build a speech recognition system, collect our own dataset of speech from 10 speakers uttering diverse phrases and sentences, and use that dataset to show how effective the system is.

This is the initial planning version. The repository currently contains:
- the project plan
- the data-collection protocol and consent form
- the draft recording prompts
- the planned folder structure
- the planning poster

The recogniser, the dataset, the experiments and the demo are planned for the coming weeks.

## Problem Statement

The project brief:

> *Create a speech recognition system. Gather a dataset consisting of speech samples from 10 speakers uttering diverse phrases or sentences. Showcase the effectiveness of your method using this dataset.*

Speech recognisers usually work well on the voices they were trained on but make far more mistakes on:
- unfamiliar accents
- local names
- noisy recordings

Bangladeshi speakers reading English sentences that mention places like *Mohakhali* or *Sylhet* are a realistic example of all three.

The problem is to design a speech recognition system that converts spoken English from **new speakers** into accurate text, and to prove its effectiveness on a dataset we collect ourselves. A result only counts if it holds for voices and sentences the system has never seen.

## Why Whisper with LoRA Adaptation Is Suitable

Speech recognition maps a variable-length audio signal to a sequence of words. Whisper is an encoder–decoder transformer pre-trained on 680,000 hours of labelled speech, so it already recognises open-vocabulary English well. Fine-tuning it with **LoRA** (low-rank adapters) trains only about 1–2% of its weights. That is exactly what a small dataset of 10 speakers can support.

A pre-trained model with lightweight adaptation gives us a structured way to handle:

- accents that are rare in the model's training data,
- local names and places that the model does not expect,
- background noise (through noise augmentation during fine-tuning),
- the small size of our dataset, since few trainable weights mean less overfitting,
- limited computing power, since it fits a free Colab GPU or a laptop.

A classical recogniser (MFCC features + dynamic time warping) is included as a baseline to show what older template-matching methods can and cannot do.

## Planned Speech Recognition Formulation

### Input

Audio from a microphone or a WAV file, at any sample rate.

### Pre-processing

Audio is converted to mono 16 kHz and loudness-normalised. Silero voice activity detection trims silence before and after speech.

### Recognition Model

Whisper-small (encoder–decoder transformer) converts the audio into text. It is compared with four baselines:
- MFCC + DTW template matching
- Vosk (Kaldi hybrid HMM-DNN)
- wav2vec2-CTC
- Whisper used zero-shot, without adaptation

### Adaptation

LoRA adapters are fine-tuned on the training speakers only. Training audio is augmented with real recorded background noise, speed changes and volume changes.

### Decoding

Beam search, guided by a fixed lexicon of Bangladeshi districts, places and names. The lexicon is built from public lists before any testing.

### Output

A normalised transcript. In command mode, the system returns the closest of 15 known voice commands, or "unknown".

### Evaluation

- **Protocol:** leave-2-speakers-out 5-fold cross-validation. Every speaker is tested once, and always by a model that never heard them.
- **Headline result:** test speakers reading sentences that never appear in training.
- **Metrics:** word error rate (main metric), character error rate, command accuracy and real-time factor.
- **Statistics:** 95% confidence intervals and significance tests.

## Planned Dataset Elements

- **Voice commands:** 15 short commands, each recorded twice (e.g. "Turn off the fan.").
- **Numbers:** 10 sentences with phone numbers, times, money and dates.
- **Phonetically rich sentences:** 15 Harvard sentences that cover the sounds of English.
- **Everyday sentences and questions:** 10 conversational sentences.
- **Bangladeshi names and places:** 10 sentences such as "I took a bus from Mohakhali to Sylhet."
- **Unique sentences:** 30 per speaker (300 in total), never repeated across speakers.
- **Spontaneous answers:** 5 short unscripted answers to simple questions.
- **Noisy recordings:** 10 prompts re-recorded in a real noisy place.

That is 120 recordings per speaker and about **1,200 utterances (~70 minutes)** from 10 consenting speakers (target: 5 female, 5 male). Half of each shared sentence category is held out from training so that the test sentences stay unseen.

Draft prompts: [data/prompts/prompts_v0.csv](data/prompts/prompts_v0.csv). Recording rules: [docs/DATA_COLLECTION_PROTOCOL.md](docs/DATA_COLLECTION_PROTOCOL.md).

## Repository Structure

```text
TenVoices-Speech-Recognition/
|-- main.py                  (planned) transcribe / evaluate / reproduce
|-- app.py                   (planned) Gradio live demo
|-- README.md
|-- requirements.txt
|-- LICENSE
|-- .gitignore
|-- configs/                 config.yaml, lexicon_bd.txt
|-- support/
|   |-- config.py            Member 1
|   |-- audio.py             Member 1
|   |-- dataset.py           Member 1
|   |-- asr.py               Member 2
|   |-- decoding.py          Member 2
|   |-- finetune.py          Member 2
|   |-- augment.py           Member 3
|   |-- visualization.py     Member 3
|   |-- metrics.py           Member 4
|   |-- dtw.py               Member 4
|   |-- commands.py          Member 4
|   `-- experiments.py       Member 4
|-- tools/                   recorder.py, qc_report.py, make_splits.py
|-- notebooks/               Colab notebook for LoRA fine-tuning
|-- tests/
|-- data/
|   |-- README.md
|   |-- prompts/             prompts_v0.csv
|   |-- metadata/            speakers and utterances templates
|   |-- splits/
|   `-- raw/, processed/     local audio only (gitignored)
|-- docs/
|   |-- PROJECT_PLAN.md
|   |-- DATA_COLLECTION_PROTOCOL.md
|   |-- consent_form.md
|   |-- templates/           individual update report template
|   `-- logs/                weekly progress log per member
|-- poster/poster.html       poster source
|-- images/                  poster and committed figures
|-- outputs/                 generated results (gitignored)
`-- others/                  reports, slides, demo video
```

## Current Progress

- Studied our Group 5 MDP project and the Group 6 speech project, and wrote down what to reuse and what to avoid.
- Wrote the full project plan: method, evaluation protocol, roles, roadmap and risks ([docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)).
- Wrote the data-collection protocol and the speaker consent form.
- Drafted 60 shared recording prompts and 5 spontaneous topics.
- Defined metadata templates for speakers and utterances.
- Created the folder structure, `.gitignore`, licence and the planning poster.

## Requirements

The planning version has no code yet. The following libraries are listed in `requirements.txt` because they are expected to be needed in later stages:

- `numpy`, `pandas`, `scipy`, `matplotlib`
- `librosa`, `soundfile`, `pyloudnorm`, `silero-vad`, `audiomentations`
- `torch`, `torchaudio`, `transformers`, `peft`, `accelerate`
- `jiwer`, `rapidfuzz`
- `gradio`

## Planned Development Roadmap

- Week 1: Planning, repository setup, recording tool and pilot recordings.
- Week 2: Dataset collection from 10 speakers, quality control and transcripts.
- Week 3: Pre-processing, baseline recognisers and evaluation metrics.
- Week 4: Our method: domain-aware decoding, noise augmentation and LoRA fine-tuning.
- Week 5: Speaker-independent experiments, noise robustness, error analysis and demo app.
- Week 6: Demo video, report, slides, poster and final polish.

The weekly tasks of each member are listed in [docs/PROJECT_PLAN.md §11](docs/PROJECT_PLAN.md#11-development-roadmap).

## Planned Work by Member

### Member 1 Data Pipeline

- Repository setup, branches and project board
- Configuration system
- Audio loading, resampling, loudness normalisation and VAD trimming
- Quality-control report for recordings
- Dataset loader and metadata manifest
- Speaker-independent fold generation with leakage tests
- `main.py` integration

### Member 2 ASR Models

- Common recogniser interface
- Whisper, wav2vec2 and Vosk recognisers
- Domain-aware decoding with the Bangladeshi lexicon
- LoRA fine-tuning notebook for all 5 folds
- Ablation runs and speed benchmark

### Member 3 Data Collection and Demo

- Recorder tool for recording sessions
- Recording protocol, consent form and volunteer scheduling
- Noise bank and noise augmentation
- Result figures (WER charts, per-speaker heatmap, confusion matrix)
- Gradio demo app, poster and demo video

### Member 4 Evaluation and Experiments

- WER, CER and sentence-error metrics with text normalisation
- MFCC + DTW classical baseline
- Command matching for command mode
- Experiment runner over all folds and subsets
- Confidence intervals and significance tests
- Results tables and error analysis

All four members record themselves, bring volunteer speakers, and check another member's recordings.

## How to Run Current Version

The current version is planning only and has no runnable code yet. Instructions will be added here once `main.py` and the recorder tool work.

To start working on your part:

```bash
git clone <this-repo-url>
cd TenVoices-Speech-Recognition
git checkout -b member<N>-<area>    # member1-data | member2-asr | member3-collection | member4-evaluation
```

## Future Outputs

The following outputs are planned for later weeks:

- `outputs/zero_shot_results.csv` and `outputs/main_results.csv`
- `images/wer_comparison.png`, `images/per_speaker_wer.png`, `images/wer_vs_snr.png`, `images/command_confusion.png`
- `tenvoices-v1.0.zip`: the dataset release (consenting speakers, CC BY 4.0)
- `others/final_report.pdf`, the presentation slides and `others/demo_video.mp4`
- The results edition of the poster

These files are not generated in the planning version.

## Planning Documents

| Document | Contents |
| --- | --- |
| [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Interpretation of the brief, lessons from earlier projects, method, experiments, evaluation protocol, roles, weekly roadmap, risks, ethics |
| [docs/DATA_COLLECTION_PROTOCOL.md](docs/DATA_COLLECTION_PROTOCOL.md) | Session script, recording settings, file naming, metadata, transcription rules, quality control, privacy |
| [docs/consent_form.md](docs/consent_form.md) | Speaker consent form with an opt-in for public release |
| [docs/templates/individual_update_report.md](docs/templates/individual_update_report.md) | Template for the individual update reports |

## Learning from Previous Projects

- **[Autonomous Volcanic Terrain Exploration Using MDP](https://github.com/MRSHAKILS/Autonomous-Volcanic-Terrain-Exploration-Using-Markov-Decision-Process)** (our Group 5 project). We reuse:
  - the `support/` module layout with one owner per file
  - one branch per member
  - seeded, reproducible experiments
  - baseline comparisons
  - the same set of deliverables
- **[Speech Accent Recognition, Group 6](https://github.com/alvi00/speech_recognization_systems-group-6)**. This was accent classification on a public dataset. We keep its classical-vs-pretrained comparison and noise augmentation. We do things differently in four ways:
  - We collect our own data.
  - We do actual speech-to-text.
  - We use speaker-independent cross-validation instead of one random split.
  - We keep the code in shared, configurable modules.

## Ethics and Privacy

- **Consent:** every speaker signs a consent form; public release is a separate opt-in.
- **Pseudonymity:** speakers are identified only as S01–S10. No names, phone numbers or student IDs are stored.
- **Storage:** raw audio stays in a private team folder and is never committed to this repository.

## License

- **Code:** [MIT](LICENSE).
- **TenVoices audio and transcripts:** CC BY 4.0, for speakers who opt in, once released.
- **Prompt sources:** Harvard sentences (IEEE 1969) and Mozilla Common Voice sentences (CC0).
