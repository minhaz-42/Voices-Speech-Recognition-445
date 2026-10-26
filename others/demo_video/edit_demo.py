"""Cut the processing waits out of the recorded demo, lay the recordings on the soundtrack, export MP4 (T27).

Usage (after ``record_demo.py``)::

    python others/demo_video/edit_demo.py outputs/demo_video <recordings folder> tenvoices_demo.mp4

Each processing wait (Whisper on a CPU takes 10-25 s) is cut to about 1.5 s; the video
says so in a caption. The recordings are loudness-normalised and placed at the moments
they were uploaded. Uses the ffmpeg that comes with ``imageio-ffmpeg``.
"""

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

raw, rec, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ev = json.loads((raw / "events.json").read_text())
ff = imageio_ffmpeg.get_ffmpeg_exe()
segs = [(ev["start"] - 0.2, ev["click_lights"] + 1.5), (ev["done_lights"] - 0.2, ev["click_music"] + 1.5),
        (ev["done_music"] - 0.2, ev["end"] + 0.5)]
lens = [b - a for a, b in segs]
total = sum(lens)
off_l = ev["lights"] - segs[0][0]
off_m = lens[0] + (ev["music"] - segs[1][0])
print(f"total {total:.1f} s; voice at {off_l:.2f} and {off_m:.2f} s", flush=True)
vf = "".join(f"[0:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS[v{i}];" for i, (a, b) in enumerate(segs))
vf += "".join(f"[v{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[vout];"
af = (f"[1:a]aresample=48000,loudnorm=I=-16:TP=-1.5,aresample=48000,adelay={int(off_l * 1000)}:all=1[a1];"
      f"[2:a]aresample=48000,loudnorm=I=-16:TP=-1.5,aresample=48000,adelay={int(off_m * 1000)}:all=1[a2];"
      f"[a1][a2]amix=inputs=2:normalize=0:duration=longest,apad=whole_dur={total:.3f},atrim=0:{total:.3f}[aout]")
subprocess.run([ff, "-y", "-loglevel", "error", "-i", ev["video"], "-i", str(rec / "lights.wav"), "-i", str(rec / "music.wav"),
                "-filter_complex", vf + af, "-map", "[vout]", "-map", "[aout]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p", "-r", "25",
                "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-movflags", "+faststart", str(out)], check=True, timeout=400)
print(out, f"{out.stat().st_size / 1e6:.2f} MB")
