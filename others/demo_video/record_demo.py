"""Record the TenVoices demo app (``python app.py --whisper``) as a captioned screen video (T27).

Drives the real app in Chrome with Playwright, uploads two recordings of a team member's
voice (``lights.wav``: "turn the lights on", ``music.wav``: "turn off the music"), and
writes the raw browser video plus ``events.json`` with the moments each recording was
uploaded and each processing wait started and ended. ``edit_demo.py`` then cuts the waits
and lays the recordings on the soundtrack.

Needs Playwright, which is not in requirements.txt (only this script uses it)::

    pip install playwright && python -m playwright install ffmpeg
    python app.py --whisper --port 7863                      # in another terminal
    python others/demo_video/record_demo.py http://127.0.0.1:7863/ outputs/demo_video <recordings folder>

Chrome must be installed. It runs as a normal window placed off-screen, because headless
recording can show half-repainted frames after scrolling.
"""

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL, OUT, REC = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
W, H = 1280, 720

CAPTION_JS = """
(text) => {
  let bar = document.getElementById('demo-caption');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'demo-caption';
    Object.assign(bar.style, {position:'fixed', left:'50%', bottom:'26px', transform:'translateX(-50%)',
      maxWidth:'1080px', background:'rgba(17,24,39,.92)', color:'#fff', padding:'12px 22px',
      borderRadius:'12px', font:'600 21px/1.35 "Segoe UI", Arial, sans-serif', textAlign:'center',
      zIndex:99999, boxShadow:'0 8px 30px rgba(0,0,0,.35)', transition:'opacity .25s'});
    document.body.appendChild(bar);
  }
  bar.style.opacity = text ? '1' : '0';
  if (text) bar.innerHTML = text;
}
"""


def main():
    events = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=False, args=["--window-position=-3000,0", "--window-size=1300,820"])
        ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1,
                                  record_video_dir=str(OUT), record_video_size={"width": W, "height": H})
        t0 = time.monotonic()
        mark = lambda name: events.__setitem__(name, round(time.monotonic() - t0, 3))
        page = ctx.new_page()
        page.goto(URL)
        page.wait_for_selector("text=Analyse")
        page.wait_for_timeout(1500)
        cap = lambda text: page.evaluate(CAPTION_JS, text)
        scroll = lambda y: page.evaluate(f"window.scrollTo({{top:{y}, behavior:'instant'}})")

        def center(locator):
            locator.evaluate("el => el.scrollIntoView({block: 'center', behavior: 'instant'})")
            page.wait_for_timeout(900)

        def outline(locator, on=True):
            locator.evaluate("(el, on) => { el.style.outline = on ? '4px solid #f97316' : '';"
                             " el.style.outlineOffset = on ? '3px' : ''; }", on)

        def analyse_clip(name, expect_text):
            page.get_by_role("button", name="Upload file").click()
            page.wait_for_timeout(400)
            page.get_by_test_id("waveform-Your phrase").get_by_test_id("file-upload").set_input_files(str(REC / f"{name}.wav"))
            mark(name)  # the recording is heard from here
            page.wait_for_timeout(4800)
            btn = page.get_by_role("button", name="Analyse")
            outline(btn)
            page.wait_for_timeout(600)
            btn.click()
            mark(f"click_{name}")  # the processing wait from here is cut in the edit
            outline(btn, False)
            cap("Processing&hellip; <span style='font-weight:400'>(waiting time shortened in this video)</span>")
            page.wait_for_function("(t) => [...document.querySelectorAll('textarea')]"
                                   ".some(x => x.value.includes(t))", arg=expect_text, timeout=120000)
            mark(f"done_{name}")
            page.wait_for_timeout(800)

        mark("start")
        cap("TenVoices &middot; speech recognition demo<br>"
            "<span style='font-weight:400'>It recognises which of 50 smart-home commands was spoken</span>")
        page.wait_for_timeout(5000)

        cap("A team member&rsquo;s own recording, through a laptop microphone: <i>&ldquo;Turn the lights on&rdquo;</i>")
        analyse_clip("lights", "nearest phrase FSC41")

        cap("Analyse: the clip is pre-processed exactly like the training data, then recognised")
        page.wait_for_timeout(2500)

        cap("What the model receives: waveform, spectrogram, log-mel spectrogram and MFCCs")
        scroll(880)
        page.wait_for_timeout(4200)
        scroll(1300)
        page.wait_for_timeout(3500)

        whisper_box = page.locator("label:has-text('Whisper transcript')").locator("xpath=..")
        center(whisper_box)
        outline(whisper_box)
        cap("Whisper, zero-shot: <i>&ldquo;Turn the lights on.&rdquo;</i> &rarr; FSC41, the correct command")
        page.wait_for_timeout(5000)
        outline(whisper_box, False)

        table = page.get_by_role("grid").first
        center(table)
        outline(table)
        cap("Our SVM, trained on only 10 speakers, does not recognise a new voice &mdash;<br>"
            "<span style='font-weight:400'>the speaker dependence our results show (22.3% on unseen speakers)</span>")
        page.wait_for_timeout(6500)
        outline(table, False)

        scroll(0)
        page.get_by_role("button", name="Clear").first.click()
        page.wait_for_timeout(700)
        cap("A second command: <i>&ldquo;Turn off the music&rdquo;</i>")
        analyse_clip("music", "nearest phrase FSC35")
        center(whisper_box)
        outline(whisper_box)
        cap("Whisper again finds the right command: FSC35, <i>&ldquo;turn off the music&rdquo;</i>")
        page.wait_for_timeout(4500)
        outline(whisper_box, False)

        page.get_by_text("The 50 phrases (TenVoices-FSC)").click()
        page.wait_for_timeout(600)
        page.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior:'instant'})")
        cap("All 50 phrases the system can recognise")
        page.wait_for_timeout(4500)

        cap("Code, results and figures: github.com/minhaz-42/Voices-Speech-Recognition-445")
        page.wait_for_timeout(4000)
        mark("end")
        video = page.video
        ctx.close()
        Path(OUT / "events.json").write_text(json.dumps({**events, "video": str(video.path())}), encoding="utf-8")
        browser.close()
    print(json.dumps(events))


if __name__ == "__main__":
    main()
