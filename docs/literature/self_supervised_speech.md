# Literature notes: self-supervised speech models

Background for report section 2 (related work), deep-learning side, and for explaining the untrained wav2vec 2.0 baseline (T18) and the linear probe (T46). Task T38, M4 · Shafil.

Every reference below was checked against its arXiv page (title, authors, year) and, where it has one, the publisher's page or DOI record. Whisper is covered by M1; MFCC and classical classifiers by M3 (T37).

## Papers

##### wav2vec 2.0 (Baevski, 2020)
- **Reference:** A. Baevski, Y. Zhou, A. Mohamed, M. Auli. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. https://arxiv.org/abs/2006.11477
- **What they did:** A convolutional encoder turns raw audio into latent frames, part of them is masked, and a Transformer learns to pick the true quantised latent for each masked frame among distractors (a contrastive task), using unlabelled audio only. Fine-tuned on LibriSpeech, it reaches 1.8 / 3.3 WER (clean / other) with all labels, and still 4.8 / 8.2 with only ten minutes of labelled speech after pre-training on 53k hours.
- **What we take from it:** It is our main model (`facebook/wav2vec2-base`): with only 7 training speakers per fold and 1,000 clips, we depend on what it learnt without labels. The ten-minute result is why we expect fine-tuning to work on so little data, and why we freeze the CNN feature encoder during fine-tuning, as the authors do.
