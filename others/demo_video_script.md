# Demo video script (T27, 1 minute)

A screen recording of the demo app, with your voice. Record with the Windows **Snipping Tool → Record** (or Xbox Game Bar, `Win + Alt + R`), 1920×1080, microphone on. Keep it under 70 s. Upload the video to the GitHub Release, not to git (see `README.md` in this folder).

## Before recording

1. In the project folder: `.\venv\Scripts\activate`
2. `python app.py --whisper` (starts the demo with the Whisper transcript ticked; the SVM gives the top-5 list)
3. Open http://127.0.0.1:7860 in Chrome, zoom to 90 % so the whole page fits.
4. Do one dry run, so Whisper is already downloaded and loaded.

## Shots

| Time | On screen | Say |
|---|---|---|
| 0–8 s | The app's start page | "This is TenVoices, our speech recognition demo. It recognises which of 50 smart-home commands was spoken." |
| 8–20 s | Click the microphone, say **"Turn the lights on"**, stop, click **Analyse** | "I record a command with the laptop microphone." |
| 20–32 s | Scroll to the four-panel figure | "This is what the model receives after pre-processing: the waveform, the spectrogram, the log-mel spectrogram and the MFCCs." |
| 32–45 s | The Whisper transcript and the top-5 table | "Zero-shot Whisper transcribes it and finds the right command. The SVM trained on only ten speakers does not recognise a new voice: that is exactly the speaker-dependence our results show." |
| 45–55 s | Record **"Turn off the music"**, Analyse | "A second command…" |
| 55–60 s | Open the phrase list accordion | "All 50 phrases are listed here. The code, results and figures are on our GitHub." |

## After recording

- Trim the start and end (Clipchamp, built into Windows), export as MP4 (1080p).
- Name it `tenvoices_demo.mp4`; give it to Tanvir for the GitHub Release (T28), or attach it yourself to the release draft.
- Mark T27 ✅ on `docs/team/M4_shafil.md`.
