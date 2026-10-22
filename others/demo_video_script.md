# Demo video: step-by-step guide (T27, about 1 minute)

A screen recording of the demo app with your voice, recorded on Windows 11 with the built-in **Snipping Tool** and trimmed in **Clipchamp**. Plan about 30 minutes, including two practice runs.

## 1. Prepare (5 minutes)

1. Close other apps, turn off notifications (**Settings → System → Notifications → Do not disturb**) and pick a quiet room. Plug in the laptop.
2. Open VS Code in `E:\CSE445_V\Voices-Speech-Recognition-445` and a terminal (**Terminal → New Terminal**):
   ```
   .\venv\Scripts\activate
   python app.py --whisper
   ```
   Wait for `Running on local URL: http://127.0.0.1:7860`. Leave the terminal open; closing it stops the app.
3. Open **http://127.0.0.1:7860** in Chrome. Press **F11** for full screen and **Ctrl + minus** until the page fits the screen (about 80–90 %).
4. **Warm-up run (important):** click the microphone in **Your phrase**, allow microphone access, say "turn the lights on", stop, click **Analyse**. The first Whisper run takes 10–30 seconds to load; afterwards it is fast. Then click the **×** on the audio box to clear it.

## 2. Practise once without recording

Go through the shots below once, reading the narration aloud, and check you stay around 60 seconds. The SVM's top-5 list will usually be wrong for your voice; that is expected and part of the story (the narration explains it).

## 3. Shots and narration

| Time | What you do | What you say |
|---|---|---|
| 0–8 s | Show the app's start page, scroll slowly to the top. | "This is TenVoices, our speech recognition demo. It recognises which of 50 smart-home commands was spoken." |
| 8–18 s | Click the microphone, say **"Turn the lights on"**, click stop, click **Analyse**. | (Only the command, then:) "I record a command with the laptop microphone." |
| 18–30 s | Scroll down to the four-panel figure; point with the mouse at each panel. | "This is what the model receives after pre-processing: the waveform, the spectrogram, the log-mel spectrogram and the MFCC features." |
| 30–45 s | Scroll up to **Whisper transcript**, then to the top-5 table. | "Whisper, used zero-shot, transcribes it correctly and finds the right command. Our SVM, trained on only ten speakers, does not recognise a new voice: exactly the speaker dependence our results show." |
| 45–55 s | Clear the audio (**×**), record **"Turn off the music"**, click **Analyse**, show the Whisper line. | "A second command, again recognised by Whisper." |
| 55–60 s | Open **The 50 phrases** at the bottom. | "All 50 phrases are listed here. The code, results and figures are on our GitHub." |

## 4. Record

1. Open **Snipping Tool** (Start menu), click the **video camera** icon (Record), then **+ New**.
2. Drag a box around the whole Chrome window (or the full screen).
3. Before starting, turn **on** the **microphone** button in the recording bar, so your narration is recorded. System audio can stay off.
4. Click **Start**, wait for the 3-second countdown, and follow the shots.
5. Click **Stop** (red square). The recording opens in Snipping Tool: click **Save** (floppy disk), save as `tenvoices_demo_raw.mp4` in `E:\CSE445_V\recordings\` (outside the project folder).

If something goes wrong, record again; a clean second take is quicker than editing.

(Alternative: **Xbox Game Bar**, `Win + G`, Capture → turn the microphone on → Record, or `Win + Alt + R` to start/stop. It records only the active window and saves to `Videos\Captures`.)

## 5. Trim and export

1. Open **Clipchamp** (Start menu) → **Create a new video** → drag in `tenvoices_demo_raw.mp4` → drag it onto the timeline.
2. Cut the start and end: select the clip, move the playhead, press **S** to split, select the unwanted part, press **Delete**. Aim for 55–70 seconds.
3. Optional: **Text** → add a title at the start, "TenVoices · speech recognition demo".
4. **Export** → **1080p** → save as `tenvoices_demo.mp4` in `E:\CSE445_V\recordings\`.
5. Watch it once: is your voice clear, can the text be read, is it under 70 seconds?

## 6. Hand it in

- **Do not commit the video to git** (`others/README.md`: large media go to the GitHub Release). Upload it to the team Drive (or send it to Tanvir) for the release in T28.
- On `docs/team/M4_shafil.md`, set T27 to ✅ with the date, add one faculty-update line ("I recorded the 1-minute demo video: the app recognises spoken commands live with Whisper and shows what the model receives."), add a short handoff entry, then commit and push:
  ```
  git add docs/team/M4_shafil.md
  git commit -m "T27: demo video recorded (on the team Drive)"
  git push origin dev_shafil
  ```
  The push updates pull request #2 by itself.
- Stop the app: click the terminal and press **Ctrl + C**.
