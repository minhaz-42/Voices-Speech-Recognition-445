# others/

Final deliverables that are not code:

- `poster.html`: the poster source (**T39**, **T27**, M4); its rendered image is `images/poster.png`, shown at the top of the README,
- the report (**T25**, M1),
- the slides and the 1-minute demo video (**T27**, M4).

Large media (the video, high-resolution exports) are attached to the GitHub Release instead of being committed.

<!-- numbers checklist: written by python -m support.presentation -->

### Numbers checklist (generated, do not edit by hand)

**poster_results.html → images/poster_results.png**

| Placeholder | Value | Source |
|---|---|---|
| `fsc:svm:unseen:accuracy` | 22.3% | `results/svm/fsc/summary.json` |
| `fsc:w2v2_untrained:unseen:accuracy` | 2.8% | `results/w2v2_untrained/fsc/summary.json` |
| `fsc:xgboost:unseen:accuracy` | 16.9% | `results/xgboost/fsc/summary.json` |
| `fsc:knn:unseen:accuracy` | TBD | not merged yet |
| `fsc:random_forest:unseen:accuracy` | 20.9% | `results/M1_tanvir/random_forest/fsc/summary.json` |
| `fsc:w2v2_3ep:unseen:accuracy` | TBD | not merged yet |
| `fsc:w2v2_8ep_noise:unseen:accuracy` | 21.0% | `results/w2v2_8ep_noise/fsc/summary.json` |
| `fsc:wavlm_15ep_noise:unseen:accuracy` | 33.6% | `results/wavlm_15ep_noise/fsc/summary.json` |
| `fsc:hubert_15ep_noise:unseen:accuracy` | TBD | not merged yet |
| `fsc:w2v2_probe:unseen:accuracy` | 60.5% | `results/w2v2_probe/fsc/summary.json` |
| `fsc:w2v2_15ep_noise:unseen:accuracy` | 88.5% | `results/M1_tanvir/w2v2_15ep_noise/fsc/summary.json` |
| `fsc:w2v2_15ep:unseen:accuracy` | 92.8% | `results/M1_tanvir/w2v2_15ep/fsc/summary.json` |
| `fsc:whisper_small:unseen:accuracy` | 98.7% | `results/M1_tanvir/whisper_small/fsc/summary.json` |
| `fsc:svm:random:accuracy` | 30.5% | `results/svm/fsc/summary.json` |
| `fsc:random_forest:random:accuracy` | 31.0% | `results/M1_tanvir/random_forest/fsc/summary.json` |
| `fsc:w2v2_probe:random:accuracy` | 69.5% | `results/w2v2_probe/fsc/summary.json` |
| `fsc:w2v2_15ep_noise:random:accuracy` | 100.0% | `results/M1_tanvir/w2v2_15ep_noise/fsc/summary.json` |
| `fsc:whisper_small:random:accuracy` | 99.0% | `results/M1_tanvir/whisper_small/fsc/summary.json` |
| `fsc:noise:w2v2_15ep_noise:clean` | 88.5% | `results/M1_tanvir/noise/w2v2_15ep_noise_fsc.csv` |
| `fsc:noise:w2v2_15ep_noise:10` | 82.1% | `results/M1_tanvir/noise/w2v2_15ep_noise_fsc.csv` |
| `fsc:noise:w2v2_15ep_noise:5` | 74.5% | `results/M1_tanvir/noise/w2v2_15ep_noise_fsc.csv` |
| `fsc:noise:w2v2_15ep:clean` | 92.8% | `results/M1_tanvir/noise/w2v2_15ep_fsc.csv` |
| `fsc:noise:w2v2_15ep:10` | 65.5% | `results/M1_tanvir/noise/w2v2_15ep_fsc.csv` |
| `fsc:noise:w2v2_15ep:5` | 30.7% | `results/M1_tanvir/noise/w2v2_15ep_fsc.csv` |
| `fsc:noise:random_forest:clean` | 20.9% | `results/M1_tanvir/noise/random_forest_fsc.csv` |
| `fsc:noise:random_forest:10` | 2.3% | `results/M1_tanvir/noise/random_forest_fsc.csv` |
| `fsc:noise:random_forest:5` | 2.2% | `results/M1_tanvir/noise/random_forest_fsc.csv` |
| `l2arctic:random_forest:unseen:accuracy` | 17.0% | `results/M1_tanvir/random_forest/l2arctic/summary.json` |
| `l2arctic:whisper_small:unseen:accuracy` | 99.6% | `results/M1_tanvir/whisper_small/l2arctic/summary.json` |
| `fsc:wer:whisper_small` | 3.9% | `results/M1_tanvir/whisper_small/fsc/summary.json` |
| `l2arctic:wer:whisper_small` | 10.0% | `results/M1_tanvir/whisper_small/l2arctic/summary.json` |
| `demo:svm` | 0/3 | `results/demo_check/shafil_voice_fsc.csv` |
| `demo:w2v2_probe` | 0/3 | `results/demo_check/shafil_voice_fsc.csv` |
| `demo:whisper_small` | 3/3 | `results/demo_check/shafil_voice_fsc.csv` |

**slides.html → others/slides.pdf**

| Placeholder | Value | Source |
|---|---|---|
| `fsc:svm:unseen:accuracy` | 22.3% | `results/svm/fsc/summary.json` |
| `fsc:w2v2_untrained:unseen:accuracy` | 2.8% | `results/w2v2_untrained/fsc/summary.json` |
| `fsc:w2v2_probe:unseen:accuracy` | 60.5% | `results/w2v2_probe/fsc/summary.json` |
| `fsc:w2v2_15ep_noise:unseen:accuracy` | 88.5% | `results/M1_tanvir/w2v2_15ep_noise/fsc/summary.json` |
| `fsc:random_forest:unseen:accuracy` | 20.9% | `results/M1_tanvir/random_forest/fsc/summary.json` |
| `fsc:whisper_small:unseen:accuracy` | 98.7% | `results/M1_tanvir/whisper_small/fsc/summary.json` |
| `fsc:noise:w2v2_15ep_noise:clean` | 88.5% | `results/M1_tanvir/noise/w2v2_15ep_noise_fsc.csv` |
| `fsc:noise:w2v2_15ep_noise:20` | 86.8% | `results/M1_tanvir/noise/w2v2_15ep_noise_fsc.csv` |
| `fsc:noise:w2v2_15ep_noise:10` | 82.1% | `results/M1_tanvir/noise/w2v2_15ep_noise_fsc.csv` |
| `fsc:noise:w2v2_15ep_noise:5` | 74.5% | `results/M1_tanvir/noise/w2v2_15ep_noise_fsc.csv` |
| `fsc:noise:w2v2_15ep:clean` | 92.8% | `results/M1_tanvir/noise/w2v2_15ep_fsc.csv` |
| `fsc:noise:w2v2_15ep:20` | 92.1% | `results/M1_tanvir/noise/w2v2_15ep_fsc.csv` |
| `fsc:noise:w2v2_15ep:10` | 65.5% | `results/M1_tanvir/noise/w2v2_15ep_fsc.csv` |
| `fsc:noise:w2v2_15ep:5` | 30.7% | `results/M1_tanvir/noise/w2v2_15ep_fsc.csv` |
| `fsc:noise:random_forest:clean` | 20.9% | `results/M1_tanvir/noise/random_forest_fsc.csv` |
| `fsc:noise:random_forest:20` | 4.9% | `results/M1_tanvir/noise/random_forest_fsc.csv` |
| `fsc:noise:random_forest:10` | 2.3% | `results/M1_tanvir/noise/random_forest_fsc.csv` |
| `fsc:noise:random_forest:5` | 2.2% | `results/M1_tanvir/noise/random_forest_fsc.csv` |
| `l2arctic:random_forest:unseen:accuracy` | 17.0% | `results/M1_tanvir/random_forest/l2arctic/summary.json` |
| `l2arctic:whisper_small:unseen:accuracy` | 99.6% | `results/M1_tanvir/whisper_small/l2arctic/summary.json` |
| `fsc:wer:whisper_small` | 3.9% | `results/M1_tanvir/whisper_small/fsc/summary.json` |
| `l2arctic:wer:whisper_small` | 10.0% | `results/M1_tanvir/whisper_small/l2arctic/summary.json` |
| `demo:svm` | 0/3 | `results/demo_check/shafil_voice_fsc.csv` |
| `demo:w2v2_probe` | 0/3 | `results/demo_check/shafil_voice_fsc.csv` |
| `demo:whisper_small` | 3/3 | `results/demo_check/shafil_voice_fsc.csv` |

<!-- end of numbers checklist -->
